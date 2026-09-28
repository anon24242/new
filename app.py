from flask import Flask, jsonify, render_template, request

from saavn import SaavnClient, SaavnError

app = Flask(__name__)
client = SaavnClient()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/search")
def search():
    query = request.args.get("q", "").strip()
    page = max(1, request.args.get("page", 1, type=int))
    limit = min(max(1, request.args.get("limit", 36, type=int)), 50)
    if not query:
        return jsonify({"status": "error", "message": "Missing search query"}), 400
    try:
        data = client.search_songs(query, page=page, limit=limit)
    except SaavnError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 502
    return jsonify(
        {
            "status": "success",
            "query": query,
            "page": page,
            "limit": limit,
            "total_results": data["total"],
            "results": data["songs"],
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
