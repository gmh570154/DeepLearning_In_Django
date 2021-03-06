"""DeepLearning_In_Django URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import viewspython manage.py makemigration
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from algorithms import views
from word2vec import views as web_views


urlpatterns = [
    path('admin/', admin.site.urls),
    path('index/', views.add_view_log),
    path('count/', views.get_all_view_logs),
    path('get_ganji_all/', web_views.get_ganji_all),
    path('count1/', views.get_all_view_logs1),
    path('count2/', views.get_all_view_logs2)
]
