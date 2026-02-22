from flask import *
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv("conv.csv")

app = Flask(__name__)

@app.route("/", methods = ["GET", "POST"])
def home():
	chat = ""
	if request.method == "POST":
		old_chat = request.form["chat"]
		qts = request.form["qts"]
		qts = qts.strip().lower()
		texts = [qts] + data["question"].str.lower().tolist()
		cv = CountVectorizer(stop_words="english")
		vector = cv.fit_transform(texts)
		cs = cosine_similarity(vector)
		score = cs[0][1:]
		data["score"] = score * 100
		result = data.sort_values(by="score", ascending=False)
		result = result[result.score > 10]
		if len(result) == 0:
			msg = "I'm sorry, I didn't quite catch that. My knowledge is specifically focused on Soham's professional portfolio. Would you like to hear about his technical skills, his latest projects, or his engineering degree?"
		else:
			ans = result.head(1)["answer"].values[0]
			msg = ans
		new_chat = "User: " + qts + "\n" + msg
		chat = old_chat + "\n" + new_chat
		return render_template("home.html", msg=msg, chat=chat.strip())
			
		
	else:
		return render_template("home.html")

#app.run(debug=True, use_reloader=True)