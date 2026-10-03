import pytest
from app.storage import BoundedStore


def test_store_and_retrieve():
    store = BoundedStore(max_size=10)
    key = store.put("hello")
    assert store.get(key) == "hello"


def test_missing_key_raises_keyerror():
    store = BoundedStore(max_size=10)
    with pytest.raises(KeyError):
        store.get("does-not-exist")


def test_eviction_drops_oldest_first():
    store = BoundedStore(max_size=3)
    keys = [store.put(f"obj{i}") for i in range(5)]
    assert len(store) == 3
    # the two oldest should be gone
    with pytest.raises(KeyError):
        store.get(keys[0])
    with pytest.raises(KeyError):
        store.get(keys[1])
    # the three most recent should remain
    for k in keys[2:]:
        assert store.get(k) is not None


def test_accessing_an_entry_protects_it_from_eviction():
    store = BoundedStore(max_size=3)
    keys = [store.put(f"obj{i}") for i in range(3)]
    store.get(keys[0])  # touch the oldest -> moves it to most-recently-used
    store.put("obj3")   # triggers eviction of the now-least-recently-used (keys[1])
    assert store.get(keys[0]) == "obj0"  # protected by the touch above
    with pytest.raises(KeyError):
        store.get(keys[1])  # evicted
