from langchain_ollama import OllamaLLM

model = OllamaLLM(model="deepseekmini")
print(model.invoke("Come up with 10 names for a song about parrots"))