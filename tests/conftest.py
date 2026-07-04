import pytest
import signal

def pytest_configure(config):
    # Set a global timeout for all tests to prevent hanging
    # We use a signal-based approach for Unix systems
    def handler(signum, frame):
        raise TimeoutError("Test timed out after 30 seconds")
    
    signal.signal(signal.SIGALRM, handler)

@pytest.fixture(autouse=True)
def timeout():
    import signal
    signal.alarm(30) # Set timeout to 30 seconds
    yield
    signal.alarm(0) # Disable alarm
