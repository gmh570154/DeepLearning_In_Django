@echo off
cd C:/Users/Administrator/PycharmProjects/django_machine_learning/DeepLearning_In_Django
python manage.py makemigrations
python manage.py migrate
python manage.py runserver 