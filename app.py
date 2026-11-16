# -----------------------------------------------
# app.py — Flask Web Server + MySQL Database
# Supports Prediction History, Duplicate Caching,
# User Verdict Feedback, App Ratings, Stats & CSV Export.
# -----------------------------------------------
import os
import csv
import io
import hashlib
import pickle
from datetime import datetime

import numpy as np
import pymysql
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")

# -----------------------------------------------
# Auto-create fake_news_db in MySQL if missing
# -----------------------------------------------
def ensure_mysql_database():
    try:
        conn = pymysql.connect(host="localhost", user="root", password="")
        with conn.cursor() as cursor:
            cursor.execute(
                "CREATE DATABASE IF NOT EXISTS fake_news_db "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
            )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Note: Could not auto-create database ({e})")

ensure_mysql_database()

# -----------------------------------------------
# Database Configuration (XAMPP / phpMyAdmin MySQL)
# -----------------------------------------------
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL",
    "mysql+pymysql://root:@localhost/fake_news_db"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 280,
    "pool_pre_ping": True,
}

db = SQLAlchemy(app)


# -----------------------------------------------
# Database Models (3 Relational Tables)
# -----------------------------------------------
class Prediction(db.Model):
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    text_hash = db.Column(db.String(64), unique=True, nullable=False, index=True)
    news_text = db.Column(db.Text, nullable=False)
    verdict = db.Column(db.String(10), nullable=False)        # "REAL" or "FAKE"
    confidence = db.Column(db.Integer, nullable=False)
    check_count = db.Column(db.Integer, default=1, nullable=False)
    reason = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    feedbacks = db.relationship("Feedback", backref="prediction", lazy=True, cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "text_hash": self.text_hash[:8],
            "news_text": self.news_text,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "check_count": self.check_count,
            "reason": self.reason,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S")
        }


class Feedback(db.Model):
    __tablename__ = "feedback"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    prediction_id = db.Column(db.Integer, db.ForeignKey("predictions.id"), nullable=False)
    is_accurate = db.Column(db.Boolean, nullable=False)
    suggested_label = db.Column(db.String(10), nullable=False)  # "REAL" or "FAKE"
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)


class Rating(db.Model):
    __tablename__ = "ratings"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    stars = db.Column(db.Integer, nullable=False)               # 1 to 5
    comment = db.Column(db.String(500), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)


with app.app_context():
    db.create_all()


# -----------------------------------------------
# Load Trained Passive Aggressive Classifier
# -----------------------------------------------
MODEL_PATH = os.path.join(BASE_DIR, "final_model.sav")

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Model loaded OK.")
else:
    model = None
    print("WARNING: final_model.sav not found. Run train_model.py first.")


def compute_text_hash(text: str) -> str:
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# -----------------------------------------------
# Routes
# -----------------------------------------------
@app.route("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True, silent=True)
    if not body or not body.get("text", "").strip():
        return jsonify({"error": "Please send some news text."}), 400

    news_text = body["text"].strip()
    text_hash = compute_text_hash(news_text)

    # 1. Check MySQL cache first for duplicate news
    existing = Prediction.query.filter_by(text_hash=text_hash).first()
    if existing:
        existing.check_count += 1
        db.session.commit()
        data = existing.to_dict()
        data["is_cached"] = True
        return jsonify(data)

    # 2. Otherwise run the ML model
    if model is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 500

    prediction_label = str(model.predict([news_text])[0]).upper()
    try:
        raw_score = model.decision_function([news_text])[0]
        confidence = int(np.clip(55 + abs(float(raw_score)) * 8, 55, 97))
    except Exception:
        confidence = 75

    new_record = Prediction(
        text_hash=text_hash,
        news_text=news_text,
        verdict=prediction_label,
        confidence=confidence,
        check_count=1,
        reason="Analysed by your trained Passive Aggressive classifier."
    )
    db.session.add(new_record)
    db.session.commit()

    data = new_record.to_dict()
    data["is_cached"] = False
    return jsonify(data)


@app.route("/history", methods=["GET"])
def get_history():
    query = Prediction.query
    search_q = (request.args.get("q") or "").strip()
    verdict_filter = (request.args.get("verdict") or "").strip().upper()

    if search_q:
        query = query.filter(Prediction.news_text.ilike(f"%{search_q}%"))
    if verdict_filter in ("REAL", "FAKE"):
        query = query.filter(Prediction.verdict == verdict_filter)

    records = query.order_by(Prediction.created_at.desc()).limit(50).all()
    return jsonify([r.to_dict() for r in records])


@app.route("/feedback", methods=["POST"])
def submit_feedback():
    body = request.get_json(force=True, silent=True) or {}
    prediction_id = body.get("prediction_id")
    is_accurate = bool(body.get("is_accurate"))

    pred = Prediction.query.get(prediction_id)
    if not pred:
        return jsonify({"error": "Prediction record not found."}), 404

    suggested = pred.verdict if is_accurate else ("FAKE" if pred.verdict == "REAL" else "REAL")

    fb = Feedback(
        prediction_id=pred.id,
        is_accurate=is_accurate,
        suggested_label=suggested
    )
    db.session.add(fb)
    db.session.commit()

    return jsonify({"message": "Feedback saved to MySQL!", "suggested_label": suggested})


@app.route("/rate", methods=["POST"])
def submit_rating():
    body = request.get_json(force=True, silent=True) or {}
    stars = int(body.get("stars", 5))
    comment = (body.get("comment") or "").strip()[:500]

    if stars < 1 or stars > 5:
        return jsonify({"error": "Stars must be between 1 and 5."}), 400

    rating = Rating(stars=stars, comment=comment)
    db.session.add(rating)
    db.session.commit()

    avg_stars, total_votes = db.session.query(
        func.avg(Rating.stars), func.count(Rating.id)
    ).first()

    return jsonify({
        "message": "Rating saved!",
        "average": round(float(avg_stars or 5.0), 1),
        "total_votes": int(total_votes or 1)
    })


@app.route("/stats", methods=["GET"])
def get_stats():
    total_unique = db.session.query(func.count(Prediction.id)).scalar() or 0
    total_checks = db.session.query(func.sum(Prediction.check_count)).scalar() or 0
    real_count = db.session.query(func.count(Prediction.id)).filter_by(verdict="REAL").scalar() or 0
    fake_count = db.session.query(func.count(Prediction.id)).filter_by(verdict="FAKE").scalar() or 0
    avg_conf = db.session.query(func.avg(Prediction.confidence)).scalar() or 0

    total_fb = db.session.query(func.count(Feedback.id)).scalar() or 0
    accurate_fb = db.session.query(func.count(Feedback.id)).filter_by(is_accurate=True).scalar() or 0
    agreement_rate = round((accurate_fb / total_fb) * 100, 1) if total_fb > 0 else 100.0

    return jsonify({
        "total_unique": int(total_unique),
        "total_checks": int(total_checks),
        "cache_hits": int(total_checks - total_unique),
        "real_count": int(real_count),
        "fake_count": int(fake_count),
        "avg_confidence": round(float(avg_conf), 1),
        "agreement_rate": agreement_rate
    })


@app.route("/history/<int:pred_id>", methods=["DELETE"])
def delete_history_item(pred_id):
    pred = Prediction.query.get(pred_id)
    if not pred:
        return jsonify({"error": "Record not found."}), 404
    db.session.delete(pred)
    db.session.commit()
    return jsonify({"message": "Record deleted."})


@app.route("/export", methods=["GET"])
def export_csv():
    records = Prediction.query.order_by(Prediction.created_at.desc()).all()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Hash", "News Text", "Verdict", "Confidence (%)", "Check Count", "Timestamp"])

    for r in records:
        writer.writerow([r.id, r.text_hash[:8], r.news_text, r.verdict, r.confidence, r.check_count, r.created_at])

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=fake_news_history.csv"}
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)