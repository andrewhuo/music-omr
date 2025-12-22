from music21 import converter

def extract_measures(musicxml_path: str):
    """
    Parse a MusicXML file and return a list of (measure_number, offset_in_score).
    """
    score = converter.parse(musicxml_path)
    measures_data = []

    for part in score.parts:
        for measure in part.getElementsByClass('Measure'):
            num = measure.measureNumber
            offset = measure.offset
            measures_data.append((num, offset))

    return measures_data

if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    measures = extract_measures(path)
    for m in measures:
        print(f"Measure {m[0]} at offset {m[1]}")
