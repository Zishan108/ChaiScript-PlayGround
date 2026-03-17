from django.db.models.signals import post_save
from django.contrib.auth.models import User
from django.dispatch import receiver
from .models import UserProfile


@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        avatar = ''
        # Pull Google avatar if available
        try:
            sa = instance.socialaccount_set.filter(provider='google').first()
            if sa:
                avatar = sa.extra_data.get('picture', '')
        except Exception:
            pass
        UserProfile.objects.get_or_create(user=instance, defaults={'avatar_url': avatar})


@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    if hasattr(instance, 'profile'):
        instance.profile.save()