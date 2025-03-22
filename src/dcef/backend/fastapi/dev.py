# main.py
import uvicorn

if __name__ == "__main__":
	uvicorn.run("src.main:app", host="0.0.0.0", port=21011, reload=True, workers=1)
