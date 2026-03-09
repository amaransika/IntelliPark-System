from backend.main import clean_and_format_plate

def test_perfect_plate_format():
    assert clean_and_format_plate("WPABC1234") == "ABC-1234"
    assert clean_and_format_plate("CPCAA5678") == "CAA-5678"

def test_ocr_number_letter_confusion_fixes():
    assert clean_and_format_plate("WPAB01Z3A") == "ABO-1234" 
    assert clean_and_format_plate("SG5888B") == "SGS-8888" 

def test_invalid_short_plates():
    assert clean_and_format_plate("CAR") == None