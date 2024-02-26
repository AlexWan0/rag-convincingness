import justext


def justext_html_to_text(html: str) ->  str:
    try:
        paragraphs = justext.justext(html, justext.get_stoplist("English"))
    except Exception as e:
        print(e)
        return None
    
    paragraphs_clean = []
    for paragraph in paragraphs:
        if not paragraph.is_boilerplate:
            paragraphs_clean.append(paragraph.text)

    return '\n'.join(paragraphs_clean)
