from flask import Flask, jsonify, request, render_template
from sqlalchemy.orm import sessionmaker
from service_repository.database import engine
from service_repository.crud import get_corpus_by_date_range

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/news')
def get_news():
    date = request.args.get('date')
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

    news = get_corpus_by_date_range(db, date, date)  # Фильтруем новости по дате
    start = (page - 1) * per_page
    end = start + per_page
    paginated_news = news[start:end]

    news_list = [{"title": n.title, "url": n.url, "date": n.post_dttm} for n in paginated_news]
    return jsonify(news_list)

if __name__ == "__main__":
    app.run(debug=True)