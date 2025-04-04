import pymongo

# Your MongoDB config
MONGO_URI = "mongodb+srv://joanroche1604:o4HMNklN8mfozRYk@cluster0.gswdx.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
MONGO_DB_NAME = "test"

# Connect and inspect
client = pymongo.MongoClient(MONGO_URI)
db = client[MONGO_DB_NAME]

print("✅ Connected to MongoDB!")

# List collections
collections = db.list_collection_names()
print("\n📂 Collections:")
for col in collections:
    print(f" - {col}")

# Show sample documents from each collection
for col in collections:
    print(f"\n🔍 Sample from '{col}':")
    sample = db[col].find_one()
    print(sample)
