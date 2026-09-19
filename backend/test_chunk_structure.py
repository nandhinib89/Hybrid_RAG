from ingestion import process_file

FILE_PATH = "data/uploads/spotify_web_app_architecture.pdf"

result = process_file(FILE_PATH)

print("\n==============================")
print("CHUNK STRUCTURE TEST")
print("==============================")

print("\nFirst chunk dictionary:")
print(result["chunks"][0])

print("\nChunk keys:")
print(result["chunks"][0].keys())