from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('dw_app.urls')),
]

# 开发环境：允许访问 /media/ 下的图片
if settings.DEBUG:
     #static() 路由只能在根 urls 配置
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
