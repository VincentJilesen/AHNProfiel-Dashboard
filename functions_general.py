"""
Bestand Algemene functies WSHD
1) Map(pen) doornemen en vinden bestanden gegeven bestandextensie(s)

@auteur Paul Notenboom WSHD
"""
import os

def find_files(infolder, extn=['']):
    """
    find_files:
    input:
        - infolder (str) = (bovenliggende) folder waarin gezocht dient te worden
        - extn (list) = extensies, altijd str vorm vb. [".gef", ".xml"]
    output:
        - lijst (dict) = {bestand : volledig pad}
    """
    # doorloop alle mappen

    lijst={}
    for root, dirs, files in os.walk(infolder, topdown=True):
        for extns in extn:
            for file in files:
                if file.endswith(extns):
                    lijst.update({file: os.path.join(root, file)})
    return lijst
