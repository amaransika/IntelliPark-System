# Python 3.11 පාවිච්චි කිරීම
FROM python:3.11

# වැඩ කරන ෆෝල්ඩරය හැදීම
WORKDIR /app

# Requirements ටික කොපි කරලා Install කිරීම
COPY ./backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# අනිත් ඔක්කොම ෆයිල්ස් කොපි කිරීම
COPY . /app

# Backend එක පණගැන්වීම (Hugging Face ඉල්ලන්නේ Port 7860)
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]