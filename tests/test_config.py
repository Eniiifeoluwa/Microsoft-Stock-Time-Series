import  pytest

class NotInRange(Exception):
    def __init__(self, message = 'Not in range at all!'):
        self.message = message
        super(NotInRange, self).__init__(self.message)



def test_generic():
    a = 5
    with pytest.raises(NotInRange):
        if a not in range(10, 15):
            raise NotInRange