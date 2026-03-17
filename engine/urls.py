from django.urls import path
from django.shortcuts import render
from . import views

def index(request):
    return render(request, "engine/index.html")

urlpatterns = [
    path("",                              index,                    name="engine_home"),
    path("run/",                          views.run_chai,           name="run_chai"),
    path("share/",                        views.share_code,         name="share_code"),
    path("share/<str:share_id>/",         views.load_shared,        name="load_shared"),

    # projects
    path("projects/save/",                views.save_project,       name="save_project"),
    path("projects/list/",                views.list_projects,      name="list_projects"),
    path("projects/<int:project_id>/",    views.load_project,       name="load_project"),

    # version history
    path("snapshots/<int:snapshot_id>/",  views.restore_snapshot,   name="restore_snapshot"),

    # profile
    path("profile/",                      views.profile,            name="profile"),

    # leaderboard
    path("leaderboard/",                  views.leaderboard,        name="leaderboard"),

    # challenges
    path("challenges/",                   views.challenge_list,     name="challenge_list"),
    path("challenges/<slug:slug>/",       views.challenge_detail,   name="challenge_detail"),
    path("challenges/<slug:slug>/submit/",views.submit_challenge,   name="submit_challenge"),
]