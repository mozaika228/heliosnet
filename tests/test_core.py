from core.node import Node


def test_node_init():
    node = Node("./config/config.example.yaml")
    assert node.config.node_id
