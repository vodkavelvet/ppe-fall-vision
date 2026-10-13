"""State machine jatuh untuk SATU orang. Murni: hanya waktu dan fitur masuk.

::

    UPRIGHT --(postur tergeletak | turun cepat)--> FALLING
    FALLING --(tergeletak bertahan persistence_s)--> FALLEN      <-- satu-satunya alert
    FALLING --(tak ada pemicu falling_timeout_s)---> UPRIGHT     (alarm palsu)
    FALLEN  --(tegak bertahan recover_s)-----------> RECOVERED   (tanpa alert)
    RECOVERED --(recovered_hold_s)-----------------> UPRIGHT

**Alert hanya saat masuk FALLEN.** Jongkok atau berlutut sepuluh detik tidak boleh
menyalakannya, karena keduanya tidak memenuhi suara postur tergeletak, dan yang
menyerupai pun tidak bertahan selama ``persistence_s``.

**Kecepatan turun hanya petunjuk, bukan syarat.** Diukur di subjek latih, puncak
kecepatan klip ADL (median 0,49) mendekati frame sekitar onset jatuh (p90 0,82).
Menjadikannya syarat FALLEN akan melewatkan jatuh pelan dan pingsan, yang justru
paling berbahaya. Kecepatan hanya mempercepat masuk ke FALLING.

**UNKNOWN tidak menggerakkan apa pun.** Frame yang posturnya tak terbaca tidak
memajukan persistence dan tidak membatalkannya. FALLEN tidak pernah dibatalkan
oleh UNKNOWN: orang yang tertutup bukan orang yang sudah bangun.

Waktu selalu dalam detik dari ``timestamp_s``, tidak pernah jumlah frame, sehingga
perilakunya sama di 20 maupun 30 fps.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from pfv.config import FallConfig, PostureConfig
from pfv.fall.features import StaticFeatures, TemporalFeatures
from pfv.fall.posture import Posture, PostureResult, classify_posture

#: Fungsi yang mengubah fitur satu frame menjadi keputusan postur.
PostureFn = Callable[[StaticFeatures, PostureConfig], PostureResult]


class FallState(str, Enum):
    UPRIGHT = "UPRIGHT"
    FALLING = "FALLING"
    FALLEN = "FALLEN"
    RECOVERED = "RECOVERED"


@dataclass(frozen=True)
class FallDecision:
    """Hasil satu langkah."""

    timestamp_s: float
    state: FallState
    previous_state: FallState
    alert: bool
    """``True`` hanya pada langkah yang masuk FALLEN dan lolos cooldown."""

    reason: str
    posture: Posture
    n_lying_votes: int
    lying_for_s: float
    """Sudah berapa lama orang ini terbaca tergeletak, 0 bila tidak."""

    onset_s: Optional[float]
    """Saat postur tergeletak PERTAMA terbaca. Acuan latensi deteksi: bandingkan
    dengan awal periode tergeletak pada ground truth, bukan dengan ``timestamp_s``."""

    @property
    def changed(self) -> bool:
        return self.state is not self.previous_state


class FallStateMachine:
    """Satu instance per ``track_id``."""

    def __init__(
        self, cfg: FallConfig, posture_fn: Optional[PostureFn] = None
    ) -> None:
        """``posture_fn`` mengganti klasifier postur bawaan.

        Ada untuk satu alasan: eksperimen ablasi Task 7b membandingkan model penuh
        dengan baseline yang umum di literatur (rasio aspek saja, sudut trunk saja)
        MELALUI state machine, persistence, dan cooldown yang SAMA. Membandingkan
        klasifier postur mentah tanpa state machine akan mengukur hal yang berbeda
        dari yang dilakukan sistem.
        """
        self._posture_cfg = cfg.posture
        self._posture_fn: PostureFn = posture_fn or classify_posture
        self._t = cfg.timing

        self.state = FallState.UPRIGHT
        self._state_since: float = 0.0
        self._last_t: Optional[float] = None

        self._lying_since: Optional[float] = None
        self._n_lying = 0
        self._n_upright = 0
        self._n_unknown = 0
        self._upright_since: Optional[float] = None
        self._unknown_since: Optional[float] = None
        self._last_trigger: float = 0.0
        self._last_alert: Optional[float] = None

    # ------------------------------------------------------------- publik

    @property
    def onset_s(self) -> Optional[float]:
        return self._lying_since

    def update(
        self,
        timestamp_s: float,
        static: StaticFeatures,
        temporal: Optional[TemporalFeatures] = None,
    ) -> FallDecision:
        if self._last_t is not None and timestamp_s < self._last_t:
            raise ValueError(
                f"timestamp mundur: {timestamp_s} setelah {self._last_t}. "
                "State machine bergantung pada urutan waktu."
            )
        self._last_t = timestamp_s

        result = self._posture_fn(static, self._posture_cfg)
        previous = self.state
        fast = self._is_fast(temporal)
        alert = False
        reason = "steady"

        if self.state is FallState.UPRIGHT:
            reason = self._step_upright(timestamp_s, result, fast)
        elif self.state is FallState.FALLING:
            reason, alert = self._step_falling(timestamp_s, result, fast)
        elif self.state is FallState.FALLEN:
            reason = self._step_fallen(timestamp_s, result)
        else:
            reason = self._step_recovered(timestamp_s, result)

        lying_for = (
            timestamp_s - self._lying_since if self._lying_since is not None else 0.0
        )
        return FallDecision(
            timestamp_s=timestamp_s,
            state=self.state,
            previous_state=previous,
            alert=alert,
            reason=reason,
            posture=result.posture,
            n_lying_votes=result.n_lying,
            lying_for_s=lying_for,
            onset_s=self._lying_since,
        )

    # ------------------------------------------------------------ internal

    def _is_fast(self, temporal: Optional[TemporalFeatures]) -> bool:
        threshold = self._t.falling_velocity
        if threshold is None or temporal is None:
            return False
        # Kecepatan yang melintasi lubang deteksi sudah dikosongkan oleh
        # compute_temporal; diperiksa lagi di sini agar tidak bergantung padanya.
        if not temporal.contiguous or temporal.hip_velocity_norm is None:
            return False
        return temporal.hip_velocity_norm >= threshold

    def _enter(self, state: FallState, t: float) -> None:
        self.state = state
        self._state_since = t

    def _begin_lying(self, t: float) -> None:
        self._lying_since = t
        self._n_lying = 0
        self._n_upright = 0
        self._n_unknown = 0
        self._upright_since = None

    def _clear_lying(self) -> None:
        self._lying_since = None
        self._n_lying = self._n_upright = self._n_unknown = 0
        self._upright_since = None

    def _cooldown_ok(self, t: float) -> bool:
        return self._last_alert is None or (t - self._last_alert) >= self._t.cooldown_s

    def _step_upright(self, t: float, result: PostureResult, fast: bool) -> str:
        if result.posture is Posture.LYING:
            self._begin_lying(t)
            self._n_lying = 1
            self._last_trigger = t
            self._unknown_since = None
            self._enter(FallState.FALLING, t)
            return "lying_posture"
        if fast:
            self._clear_lying()
            self._last_trigger = t
            self._unknown_since = None
            self._enter(FallState.FALLING, t)
            return "fast_descent"
        return "steady"

    def _step_falling(self, t: float, result: PostureResult, fast: bool):
        tm = self._t
        if fast:
            self._last_trigger = t

        if result.posture is Posture.LYING:
            if self._lying_since is None:
                self._begin_lying(t)
            self._n_lying += 1
            self._upright_since = None
            self._unknown_since = None
            self._last_trigger = t
        elif result.posture is Posture.UPRIGHT:
            self._unknown_since = None
            if self._lying_since is not None:
                self._n_upright += 1
                if self._upright_since is None:
                    self._upright_since = t
                if t - self._upright_since >= tm.grace_s:
                    # Tegak bertahan melewati toleransi: hitungan persistence batal.
                    self._clear_lying()
        else:  # UNKNOWN: tidak memajukan dan tidak membatalkan
            if self._unknown_since is None:
                self._unknown_since = t
            if self._lying_since is not None:
                self._n_unknown += 1

        if self._lying_since is not None and (t - self._lying_since) >= tm.persistence_s:
            known = self._n_lying + self._n_upright
            total = known + self._n_unknown
            lying_fraction = self._n_lying / known if known else 0.0
            known_fraction = known / total if total else 0.0
            if lying_fraction >= tm.min_lying_fraction and known_fraction >= tm.min_known_fraction:
                self._enter(FallState.FALLEN, t)
                self._upright_since = None
                if self._cooldown_ok(t):
                    self._last_alert = t
                    return "persistent_lying", True
                return "cooldown_suppressed", False
            return "persistence_evidence_weak", False

        if self._lying_since is None:
            if (t - self._last_trigger) >= tm.falling_timeout_s:
                self._enter(FallState.UPRIGHT, t)
                return "falling_timeout", False
            if self._unknown_since is not None and (t - self._unknown_since) >= tm.max_unknown_s:
                self._enter(FallState.UPRIGHT, t)
                return "unknown_timeout", False
        return "steady", False

    def _step_fallen(self, t: float, result: PostureResult) -> str:
        if result.posture is Posture.UPRIGHT:
            if self._upright_since is None:
                self._upright_since = t
            if (t - self._upright_since) >= self._t.recover_s:
                self._enter(FallState.RECOVERED, t)
                self._clear_lying()
                return "recovered"
        elif result.posture is Posture.LYING:
            self._upright_since = None
        # UNKNOWN: tidak mengubah apa pun
        return "steady"

    def _step_recovered(self, t: float, result: PostureResult) -> str:
        if result.posture is Posture.LYING:
            self._begin_lying(t)
            self._n_lying = 1
            self._last_trigger = t
            self._unknown_since = None
            self._enter(FallState.FALLING, t)
            return "relapse"
        if (t - self._state_since) >= self._t.recovered_hold_s:
            self._enter(FallState.UPRIGHT, t)
            return "recovered_hold_done"
        return "steady"
