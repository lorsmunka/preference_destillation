"""Every annealed training dial, as a function of the training step.

One cosine shape (`cosine_between`) is shared by all of them: a value starts at its
`_start` setting on step 0 and arrives at its `_end` setting on the last step. The KL
ratio can additionally oscillate (idea #36), and the learning rate has a linear warm-up.
"""

import math


class Schedules:
    def __init__(self, config: dict, steps_per_epoch: int, total_training_steps: int):
        self.steps_per_epoch = steps_per_epoch
        self.total_training_steps = total_training_steps

        self.learning_rate_peak = config['learning_rate']
        self.warmup_steps = int(total_training_steps * config['lr_warmup_ratio'])
        self.learning_rate_min = 1e-5

        self.kl_ratio_start = config['kl_ratio_start']
        self.kl_ratio_end = config['kl_ratio_end']
        self.student_temperature_start = config['distillation_temperature_start']
        self.student_temperature_end = config['distillation_temperature_end']

        # Newer dials are read with defaults so older dict configs and info.json files still load.
        self.teacher_temperature_start = config.get('teacher_temperature_start')
        self.teacher_temperature_end = config.get('teacher_temperature_end')
        if self.teacher_temperature_start is not None and self.teacher_temperature_end is None:
            self.teacher_temperature_end = self.teacher_temperature_start
        self.kl_ratio_cycle_epochs = config.get('kl_ratio_cycle_epochs', 0.0)
        self.kl_ratio_cycle_amplitude = config.get('kl_ratio_cycle_amplitude', 0.0)
        self.attention_noise_start = config.get('attention_noise_start', 0.0)
        self.attention_noise_end = config.get('attention_noise_end', 0.0)

    def cosine_between(self, start: float, end: float, step: int) -> float:
        if self.total_training_steps <= 1:
            return start
        progress = step / self.total_training_steps
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return end + (start - end) * cosine_decay

    def student_temperature(self, step: int) -> float:
        return self.cosine_between(self.student_temperature_start, self.student_temperature_end, step)

    def teacher_temperature(self, step: int) -> float:
        """The teacher's own softmax temperature — the entropy dial. Follows the student's
        temperature unless teacher_temperature_start is set."""
        if self.teacher_temperature_start is None:
            return self.student_temperature(step)
        return self.cosine_between(self.teacher_temperature_start, self.teacher_temperature_end, step)

    def kl_ratio(self, step: int) -> float:
        ratio = self.cosine_between(self.kl_ratio_start, self.kl_ratio_end, step)
        if self.kl_ratio_cycle_epochs > 0 and self.kl_ratio_cycle_amplitude > 0:
            progress_in_epochs = step / max(self.steps_per_epoch, 1)
            ratio += self.kl_ratio_cycle_amplitude * math.sin(
                2 * math.pi * progress_in_epochs / self.kl_ratio_cycle_epochs)
        return min(1.0, max(0.0, ratio))

    def attention_noise(self, step: int) -> float:
        return self.cosine_between(self.attention_noise_start, self.attention_noise_end, step)

    def learning_rate(self, step: int) -> float:
        if self.warmup_steps > 0 and step <= self.warmup_steps:
            return self.learning_rate_peak * (step / self.warmup_steps)
        if self.total_training_steps <= self.warmup_steps:
            return self.learning_rate_peak
        progress = (step - self.warmup_steps) / (self.total_training_steps - self.warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return self.learning_rate_min + (self.learning_rate_peak - self.learning_rate_min) * cosine_decay
