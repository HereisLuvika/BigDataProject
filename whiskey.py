import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import text

jaWhiskeyPath = "Dataset/japanese_whisky_review.csv"
scoWhiskeyPath = "Dataset/scotch_whisky_review.csv"

store = []
filedf = pd.read_csv(jaWhiskeyPath)
store.append(filedf)
jaWhiskeyData = pd.concat(store, ignore_index=False)

store = []
filedf = pd.read_csv(scoWhiskeyPath)
store.append(filedf)
scoWhiskeyData = pd.concat(store, ignore_index=False)

targetJaData = pd.DataFrame(jaWhiskeyData, columns=["Bottle_name", "Review_Content"])
targetScoData = pd.DataFrame(scoWhiskeyData, columns=["name", "description.1.2247."])
targetJaData.columns = ['whiskey', 'review']
targetScoData.columns = ['whiskey', 'review']
targetJaData['review'] = targetJaData['review'].astype(str)
targetScoData['review'] = targetScoData['review'].astype(str)

targetJaData = targetJaData.groupby("whiskey")["review"].apply(" ".join).reset_index()
targetScoData = targetScoData.groupby("whiskey")["review"].apply(" ".join).reset_index()

englishStopWords = list(text.ENGLISH_STOP_WORDS)
numStopWords = [str(i) for i in range(1000)]
addStopWord = [
    "absolutely", "alcohol", "amazing", "auction", "best", "better", "bit", 
    "blend", "blended", "bottle", "bottles", "bought", "bourbon", "buy", "cask",
    "complex", "definitely", "different", "don", "dram", "drink", "far",
    "finish", "flavor", "flavors", "flavour", "glass", "going", "good",
    "got", "great", "hakushu", "hibiki", "japan", "japanese", "just",
    "know", "light", "like", "little", "lottery", "love", "lovely", 
    "make", "mom", "money", "nice", "nikka", "nose", "notes", "palate", 
    "people", "price", "quite", "really", "say", "scotch", "single", "stuff", 
    "subtle", "suntory", "taste", "tasted", "tastes", "tasting", "think", "time", 
    "tried", "try", "value", "ve", "want", "water", "way", "whiskey", "whiskies", 
    "whisky", "world", "worth", "yamazaki", "year", "years", 
    "add", "age", "ago", "agree", "available", "away", "bad", 
    "bar", "beautiful", "big", "blends", "body", "certainly", "chance", "chaps", 
    "character", "comments", "complexity", "cost", "day", "days", "deep", "did", 
    "didn", "disappointed", "does", "doesn", "drams", "drinking", "drop", "easy", 
    "end", "enjoy", "enjoyable", "enjoyed", "especially", "excellent", "expect", 
    "expensive", "experience", "fan", "fantastic", "favorite", "favourite", "feel", 
    "fine", "free", "friend", "friends", "getting", "gift", "given", "glad", "goes", 
    "hard", "harmony", "having", "high", "highly", "hope", "hype", "ice", "icing", "interesting", 
    "isn", "leave", "let", "limited", "ll", "long", "look", "lot", "loved", "lucky", 
    "makes", "man", "market", "master", "maybe", "mouth", "need", "new", "night", 
    "online", "open", "opened", "overall", "overpriced", "pay", "pleasant", "point", 
    "post", "prefer", "pretty", "prices", "probably", "product", "purchase", "pure", 
    "quality", "range", "rating", "real", "recently", "recommend", "recommended", 
    "reminds", "review", "right", "said", "sale", "sample", "saying", "seriously", 
    "shot", "simply", "sip", "slightly", "small", "soft", "sold", "special", "stop", 
    "superb", "sure", "surprise", "taketsuru", "thing", "thought", "tongue", 
    "touch", "truly", "trying", "uk", "unique", "wait", "warm", "went", "whiskeys", 
    "whiskys", "white", "wish", "wow", "wrong", "yen", "yes", "yo", "yoichi", 
    "12yo", "absolute", "abv", "actually", "added", "airport", "buying", 
    "came", "cheap", "clean", "clear", "collection", "color", "completely", "couldn", 
    "couple", "dark", "decent", "delicate", "depth", "distillery", "dont", "double", 
    "doubt", "drinkable", "drinkers", "drops", "duty", "easily", "elegant", 
    "entry", "evening", "expected", "expert", "fact", "fresh", "gentle", "gets", 
    "giving", "gone", "green", "guys", "half", "happy", "haven", "head", 
    "higher", "highland", "hold", "impressed", "including", "incredible", "incredibly", 
    "instead", "intense", "islay", "kind", "label", "left", "level", "life", "liquor", 
    "list", "local", "looking", "lots", "macallan", "making", "mind", "mix", "months", 
    "nas", "near", "nearly", "non", "number", "oh", "ok", "old", "opinion", "outside", 
    "perfect", "perfectly", "picked", "place", "plenty", "pour", "present", "previous", 
    "problem", "profile", "purchased", "rate", "read", "reason", "received", "reserve", 
    "rest", "reviews", "rounded", "scotches", "scotland", "scottish", "sell", "selling", 
    "set", "shame", "shop", "short", "similar", "sipping", "sit", "slight", "soon", 
    "spot", "star", "start", "stock", "store", "super", "tell", "things", "times", 
    "tokyo", "took", "totally", "trip", "used", "version", "wasn", "weeks", 
    "won", "wouldn", "young", "yr", "2015", "70cl", "able", "adding", "affordable", "aged", "alternative", 
    "american", "appreciate", "awesome", "balvenie", "barrel", "believe", "saw", 
    "black", "blind", "blue", "box", "brand", "brands", "brilliant", "satisfying",
    "bring", "brought", "brown", "cabinet", "cheaper", "choice", "close", 
    "come", "comes", "coming", "compare", "compared", "complete", "peel", 
    "consider", "cool", "cork", "costs", "course", "cube", "damn", "scent", 
    "delicious", "delightful", "distinct", "doing", "drank", "drinker", "smell", "smells"
    "example", "expecting", "extremely", "face", "favorites", "favourites", 
    "felt", "flat", "followed", "general", "generally", "gives", "bargain",
    "glenfiddich", "glenmorangie", "god", "guess", "hands", "harsh", "example", 
    "heavy", "help", "hit", "honestly", "house", "huge", "job", "reminded",
    "kick", "lacks", "later", "mcgiff", "minutes", "miyagikyo", "priced", 
    "moment", "month", "morning", "needed", "normally", "note", "wonderfully",
    "obviously", "occasions", "offer", "oily", "opening", "outstanding", 
    "park", "penny", "person", "pick", "pleasure", "poured", "powerful", 
    "pricey", "reasonable", "refined", "round", "save", "says", "bite",
    "second", "sharp", "shelf", "sips", "smile", "stand", "started", "distilleries",
    "straight", "strong", "sublime", "surprised", "taking", "tasty", "offerings",
    "thank", "today", "understand", "unlike", "use", "usually", "date", "dinner",
    "website", "week", "wife", "wonderful", "wood", "word", "work", "orchard",
    "12yr", "2nd", "adds", "afraid", "air", "asked", "awards", "based", 
    "business", "casks", "change", "changed", "compares", "comparison", 
    "cornwall", "decided", "description", "despite", "difference", "golden",
    "discontinued", "effect", "enjoying", "exactly", "excited", "experienced", 
    "expression", "expressions", "exquisite", "extra", "fairly", "fall", 
    "feeling", "forgot", "fortunate", "forward", "fun", "gave", "gem", "stewed",
    "glasses", "guy", "heard", "hesitate", "hint", "hints", "eyes", "fast",
    "imo", "influence", "initial", "irish", "jim", "journey", "keeps", 
    "knew", "knows", "lasting", "late", "leaves", "likely", "lost", "low", 
    "lower", "matured", "medium", "mixed", "murray", "neat", "paid", 
    "palette", "points", "possibly", "process", "producing", "quid", 
    "rare", "refreshing", "restaurant", "run", "share", "simple", "texture",
    "solid", "somewhat", "son", "sorry", "spent", "speyside", "standard", 
    "stars", "statement", "statements", "stay", "style", "supply", "period",
    "terms", "thoroughly", "till", "treat", "unfortunately", "usual", 
    "versions", "wanted", "weekend", "words", "working", "write", "xmas", 
    "000", "background", "barrels", "bottled", "bottling", "bottlings", 
    "bowmore", "butts", "classic", "coating", "developing", "distilled", 
    "edition", "emerge", "european", "exclusive", "features", "pie",
    "filtered", "finally", "finished", "finishing", "gold", "glen", 
    "hogshead", "hogsheads", "initially", "latest", "maturation", "whiff",
    "nicely", "offers", "plus", "polished", "port", "powder", "refill", "relatively",
    "releases", "retail", "sea", "series", "shows", "slowly", "release", "released", 
    "spirit", "strength", "supple", "travel", "ultimately", "vintage", "yields", 
    "addition", "ardbeg", "batch", "brings", "caol", "core", 
    "cut", "early", "final", "freshly", "glen", "ila", "latest", 
    "older", "pedro", "plain", "plus", "suggestion", "virgin", "malts", 
]

stopWord = englishStopWords + numStopWords + addStopWord

jaVectorizer = CountVectorizer(max_features=10, stop_words = stopWord)
scoVectorizer = CountVectorizer(max_features=10, stop_words = stopWord)
jaVec = jaVectorizer.fit_transform(targetJaData["review"])
scoVec = scoVectorizer.fit_transform(targetScoData["review"])

jaCountpd = pd.DataFrame(jaVec.toarray(), columns = jaVectorizer.get_feature_names_out())
jaWordCount = jaCountpd.sum(axis = 0)
jaWordCount = jaWordCount.reset_index()
jaWordCount.columns = ["feature", "counted"]
jaWordCount = jaWordCount.sort_values(by = "counted", ascending = False)
jaWordCount = jaWordCount.reset_index(drop = True)

scoCountpd = pd.DataFrame(scoVec.toarray(), columns = scoVectorizer.get_feature_names_out())
scoWordCount = scoCountpd.sum(axis = 0)
scoWordCount = scoWordCount.reset_index()
scoWordCount.columns = ["feature", "counted"]
scoWordCount = scoWordCount.sort_values(by = "counted", ascending = False)
scoWordCount = scoWordCount.reset_index(drop = True)

jaWordCount['normalized'] = jaWordCount['counted'] / jaWordCount['counted'].sum() * 100
scoWordCount['normalized'] = scoWordCount['counted'] / scoWordCount['counted'].sum() * 100

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.pie(jaWordCount['normalized'], labels = jaWordCount['feature'], autopct='%.1f%%')
plt.title("Japan Whiskey feature")

plt.subplot(1, 2, 2)
plt.pie(scoWordCount['normalized'], labels = scoWordCount['feature'], autopct='%.1f%%')
plt.title("British Whiskey feature")

plt.show()