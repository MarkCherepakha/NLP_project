from spacy.language import Language

@Language.component("custom_component")
def custom_component(doc):
    doc.user_data["custom_data"] = "Додаткові дані"
    return doc