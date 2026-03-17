from django.contrib import admin
from django.urls import path, include
from django.shortcuts import render

handler404 = 'engine.views.custom_404'

def home(request):
    return render(request, "engine/home.html")

def docs(request):
    return render(request, "engine/docs.html")

def downloads(request):
    return render(request, "engine/downloads.html")

urlpatterns = [
    path("",           home,                      name="home"),
    path("admin/",     admin.site.urls),
    path("engine/",    include("engine.urls")),
    path("docs/",      docs,                      name="docs"),
    path("downloads/", downloads,                 name="downloads"),
    path("accounts/",  include("allauth.urls")),
]