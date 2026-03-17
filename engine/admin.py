from django.contrib import admin
from .models import (
    SharedCode, UserProfile, Project,
    VersionSnapshot, Challenge, ChallengeSubmission
)


@admin.register(SharedCode)
class SharedCodeAdmin(admin.ModelAdmin):
    list_display  = ('share_id', 'created')
    search_fields = ('share_id',)
    readonly_fields = ('share_id', 'created')


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display  = ('user', 'total_runs', 'created_at')
    search_fields = ('user__username',)
    readonly_fields = ('created_at',)


@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display  = ('title', 'user', 'updated_at')
    search_fields = ('title', 'user__username')
    readonly_fields = ('created_at', 'updated_at')


@admin.register(VersionSnapshot)
class VersionSnapshotAdmin(admin.ModelAdmin):
    list_display  = ('project', 'label', 'created_at')
    search_fields = ('project__title',)
    readonly_fields = ('created_at',)


@admin.register(Challenge)
class ChallengeAdmin(admin.ModelAdmin):
    list_display       = ('order', 'title', 'difficulty', 'is_active', 'slug')
    list_display_links = ('title',)
    list_editable      = ('order', 'is_active')
    list_filter   = ('difficulty', 'is_active')
    search_fields = ('title', 'slug')
    prepopulated_fields = {'slug': ('title',)}  # auto-fills slug from title
    fieldsets = (
        (None, {
            'fields': ('title', 'slug', 'difficulty', 'order', 'is_active')
        }),
        ('Content', {
            'fields': ('description', 'starter')
        }),
        ('Auto-grader (hidden from users)', {
            'fields': ('test_code',),
            'classes': ('collapse',),
        }),
    )


@admin.register(ChallengeSubmission)
class ChallengeSubmissionAdmin(admin.ModelAdmin):
    list_display  = ('user', 'challenge', 'passed', 'exec_time', 'submitted_at')
    list_filter   = ('passed', 'challenge')
    search_fields = ('user__username', 'challenge__title')
    readonly_fields = ('submitted_at',)