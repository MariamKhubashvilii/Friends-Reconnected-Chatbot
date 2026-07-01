import chromadb
import subprocess

def ram_free():
    r = subprocess.run(["vm_stat"], capture_output=True, text=True)
    lines = r.stdout.splitlines()
    page_size = 4096
    free = 0
    for l in lines:
        if "Pages free" in l:
            free = int(l.split(":")[1].strip().rstrip(".")) * page_size
    return free / 1e9

print(f"RAM free before: {ram_free():.2f} GB")

client = chromadb.PersistentClient(path="./test_chroma_db")
print("Client created")
print(f"RAM free after client: {ram_free():.2f} GB")

existing = [c.name for c in client.list_collections()]
print(f"Existing collections: {existing}")

collection = client.get_or_create_collection(name="test_collection")
print("Collection created")
print(f"RAM free after collection: {ram_free():.2f} GB")

collection.add(
    ids=["1", "2"],
    embeddings=[[0.1]*384, [0.2]*384],
    documents=["test doc one", "test doc two"]
)
print("Added test data")
print(f"RAM free after add: {ram_free():.2f} GB")
print(f"Count: {collection.count()}")
