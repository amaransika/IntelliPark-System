from backend.main import Coordinate

def test_coordinate_scaling():
    coord = Coordinate(x=10, y=20, w=100, h=50)
    assert coord.x == 10
    assert coord.w == 100
    
    sc_x = 0.5
    sc_y = 0.5
    scaled_x = int(coord.x * sc_x)
    scaled_y = int(coord.y * sc_y)
    
    assert scaled_x == 5
    assert scaled_y == 10