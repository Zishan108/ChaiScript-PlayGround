from django.contrib import admin
from django.urls import path, include
from django.shortcuts import render
from django.http import HttpResponse
from django.contrib.auth.models import User
from engine import views as engine_views

handler404 = 'engine.views.custom_404'

def docs(request):
    return render(request, "engine/docs.html")

def downloads(request):
    return render(request, "engine/downloads.html")

def create_super(request):
    if not User.objects.filter(username='kami').exists():
        User.objects.create_superuser('kami', 'zishan@example.com', 'Kami@1234')
        return HttpResponse('Superuser created!')
    return HttpResponse('Already exists!')

urlpatterns = [
    path("",           engine_views.home,         name="home"),
    path("admin/",     admin.site.urls),
    path("engine/",    include("engine.urls")),
    path("docs/",      docs,                      name="docs"),
    path("downloads/", downloads,                 name="downloads"),
    path("accounts/",  include("allauth.urls")),
    path("init-super/", create_super),
]