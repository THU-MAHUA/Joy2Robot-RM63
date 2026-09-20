"""Contact detection and force command, independent of ROS and policy weights."""
import math


class HybridForce:
    def __init__(self):
        self.filtered = 0.0
        self.count = 0
        self.contact = False
        self.fault = False
        self.received = None

    def update(self, raw_fz, now):
        if not math.isfinite(raw_fz):
            self.fault = True
            raise ValueError("Nonfinite force feedback")
        if self.received is not None:
            if now <= self.received:
                return
            if now - self.received > .25:
                self.count = 0
        compression = -float(raw_fz)
        self.filtered = .25 * compression + .75 * self.filtered
        self.received = now
        self.count = self.count + 1 if self.filtered >= 2 else 0
        self.contact |= self.count >= 3
        self.fault |= max(0, self.filtered) >= 15

    def reset_contact(self):
        # Pickup contact is not insertion contact. A force fault is never reset here.
        self.filtered = 0.0
        self.count = 0
        self.contact = False

    def commanded_compression(self):
        if not self.contact or self.fault:
            return 0.0
        return max(0.0, min(10.0, 7 + .5 * (7 - max(0, self.filtered))))
