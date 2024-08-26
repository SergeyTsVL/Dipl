## README

# Платформа распознавания объектов "Распознавание"

## Описание

Это веб-приложение на базе фреймворка Django, которое позволяет пользователям применять предобученные модели 
для обнаружения и классификации объектов на фотографиях. Приложение поддерживает механизмы регистрации и авторизации 
пользователей, что позволяет каждому пользователю создавать аккаунты и видеть только свои загруженные изображения.


## Основные функции

- Регистрация и авторизация пользователей.
- Загрузка изображений пользователями.
- Применение предобученных моделей для распознавания и классификации объектов на загруженных изображениях.
- Применение предобученных моделей для распознавания изображений лиц с встроенной видеокамеры.
- Отображение результатов распознавания.
- Возможность удаления загруженных изображений.

## Предобученные модели

### 1. MobileNet SSD

MobileNet SSD определяет следующие классы объектов:

- "aeroplane" (самолёт)
- "bicycle" (велосипед)
- "bird" (птица)
- "boat" (лодка)
- "bottle" (бутылка)
- "bus" (автобус)
- "car" (автомобиль)
- "cat" (кот)
- "chair" (стул)
- "cow" (корова)
- "diningtable" (стол)
- "dog" (собака)
- "horse" (лошадь)
- "motorbike" (мотоцикл)
- "person" (человек)
- "pottedplant" (цветок)
- "sheep" (овца)
- "sofa" (диван)
- "train" (поезд)
- "tvmonitor" (телевизор)
- "background" (фон)

### 2. BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation

BLIP используется для описания сюжета на картинке, предоставляя текстовое описание изображения.

## Установка и настройка

1. Клонируйте репозиторий:

```bash
git clone https://github.com/SergeyTsVL/Dipl.git
```

2. Создайте и активируйте виртуальное окружение:

```bash
python -m venv venv
source venv/bin/activate  # для Windows используйте `venv\Scripts\activate`
```

3. Установите зависимости:

```bash
pip install -r requirements.txt
```
Необходимо учесть при отсутствии физической возможности запуска/доступу встроенной камеры, будет ошибка⤵️⤵️⤵️
OpenCV(4.8.1) D:\a\opencv-python\opencv-python\opencv\modules\imgproc\src\color.cpp:182: error: (-215:Assertion failed) !_src.empty() in function 'cv::cvtColor'

4. Примените миграции:

```bash
cd detection_site
python manage.py makemigrations
python manage.py migrate
```

5. Создайте суперпользователя для доступа к административной панели:

```bash
python manage.py createsuperuser
```

6. Запустите сервер разработки:

```bash
python manage.py runserver
```

7. Перейдите по адресу [http://127.0.0.1:8000](http://127.0.0.1:8000) в браузере.

## Использование

1. Зарегистрируйтесь или войдите в систему.
2. Загрузите изображение для анализа.
3. Выберите модель для обработки изображения: MobileNet SSD или BLIP.
4. Получите результаты распознавания или описания сюжета изображения.
5. Удалите ненужные изображения через дашборд.
6. Включите видеокамеру с функцией распознавания лица.
7. Отключите камеру с распознаванием лица.
