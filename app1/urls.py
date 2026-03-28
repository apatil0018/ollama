from django.urls import path
from .views import QueryDoc

urlpatterns = [
    path('query/', QueryDoc.as_view(), name='query-doc'),
]