from django.urls import path
from .views import run_chai, share_code, load_shared, docs
from django.shortcuts import render

def index(request):
    return render(request, "engine/index.html")

urlpatterns = [
    path("", index, name="engine_home"),     # landing page
    path("run/", run_chai, name="run_chai"), # execute code
    path("share/", share_code, name="share_code"),
    path("share/<str:share_id>/", load_shared, name="load_shared"),
]