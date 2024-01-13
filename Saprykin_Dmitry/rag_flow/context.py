from promptflow import tool

@tool
def context(documents: object) -> str:
  return {"documents": documents}
