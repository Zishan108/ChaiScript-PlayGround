from django.urls import path
from .views import run_chai
from django.shortcuts import render

def index(request):
    return render(request, "engine/index.html")

urlpatterns = [
    path("", index, name="engine_home"),     # landing page
    path("run/", run_chai, name="run_chai"), # execute code
]