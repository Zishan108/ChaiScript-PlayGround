from django.db import models
import uuid

class SharedCode(models.Model):
    share_id = models.CharField(max_length=12, unique=True, default='')
    code     = models.TextField()
    created  = models.DateTimeField(auto_now_add=True)

    def save(self, *args, **kwargs):
        if not self.share_id:
            self.share_id = uuid.uuid4().hex[:12]
        super().save(*args, **kwargs)