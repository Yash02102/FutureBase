from src.rules.commerce import commerce_intent_classifier


def test_commerce_intent_purchase_with_entities():
    classifier = commerce_intent_classifier()
    intent = classifier.classify("Buy me a wireless headset under 5000")
    assert intent.name == "purchase"
    assert intent.entities.get("max_price") == "5000"
    assert "wireless headset" in intent.entities.get("product_query", "")


def test_commerce_intent_track_order_with_order_id():
    classifier = commerce_intent_classifier()
    intent = classifier.classify("Where is my order ord_1234")
    assert intent.name == "track_order"
    assert intent.entities.get("order_id") == "ord_1234"
