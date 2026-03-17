from django.db import models
from django.contrib.auth.models import User
import uuid


# ── Existing ──────────────────────────────────────────────
class SharedCode(models.Model):
    share_id = models.CharField(max_length=12, unique=True, default='')
    code     = models.TextField()
    created  = models.DateTimeField(auto_now_add=True)

    def save(self, *args, **kwargs):
        if not self.share_id:
            self.share_id = uuid.uuid4().hex[:12]
        super().save(*args, **kwargs)


# ── User Profile ───────────────────────────────────────────
class UserProfile(models.Model):
    user       = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    bio        = models.TextField(blank=True, default='')
    avatar_url = models.URLField(blank=True, default='')  # for Google avatar
    total_runs = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'{self.user.username} — profile'


# ── Projects ───────────────────────────────────────────────
class Project(models.Model):
    user       = models.ForeignKey(User, on_delete=models.CASCADE, related_name='projects')
    title      = models.CharField(max_length=120, default='Untitled')
    code       = models.TextField(default='')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']

    def __str__(self):
        return f'{self.user.username} / {self.title}'


# ── Version History (snapshots per project) ────────────────
class VersionSnapshot(models.Model):
    project    = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='snapshots')
    code       = models.TextField()
    label      = models.CharField(max_length=80, blank=True, default='')  # optional note
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f'{self.project.title} @ {self.created_at:%Y-%m-%d %H:%M}'


# ── Challenges ─────────────────────────────────────────────
class Challenge(models.Model):
    DIFFICULTY = [('easy', 'Easy'), ('medium', 'Medium'), ('hard', 'Hard')]

    slug        = models.SlugField(unique=True)
    title       = models.CharField(max_length=120)
    description = models.TextField()
    difficulty  = models.CharField(max_length=8, choices=DIFFICULTY, default='easy')
    starter     = models.TextField(blank=True, default='')   # starter code
    test_code   = models.TextField(default='')               # hidden test code appended on grading
    order       = models.PositiveIntegerField(default=0)
    is_active   = models.BooleanField(default=True)

    class Meta:
        ordering = ['order']

    def __str__(self):
        return self.title


# ── Challenge Submissions ──────────────────────────────────
class ChallengeSubmission(models.Model):
    user       = models.ForeignKey(User, on_delete=models.CASCADE, related_name='submissions')
    challenge  = models.ForeignKey(Challenge, on_delete=models.CASCADE, related_name='submissions')
    code       = models.TextField()
    passed     = models.BooleanField(default=False)
    output     = models.TextField(blank=True, default='')
    exec_time  = models.FloatField(default=0.0)   # seconds
    submitted_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-submitted_at']

    def __str__(self):
        status = '✓' if self.passed else '✗'
        return f'{status} {self.user.username} — {self.challenge.title}'