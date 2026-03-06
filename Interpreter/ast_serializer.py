def ast_to_json(node):
    if node is None:
        return None


    node_type = type(node).__name__

    result = {
        "type": node_type,
        "line": getattr(node.pos_start, "ln", None) + 1 if hasattr(node, "pos_start") and node.pos_start else None,
        "value": None,
        "children": []
    }

    # ---------------------------
    # CAPTURE SIMPLE VALUES
    # ---------------------------

    if hasattr(node, "tok") and node.tok:
        result["value"] = str(node.tok.value)

    if hasattr(node, "var_name_tok"):
        result["value"] = node.var_name_tok.value

    # ---------------------------
    # IMPROVED BINARY OP LABEL
    # ---------------------------

    if node_type == "BinOpNode":

        left = ast_to_json(node.left_node)
        right = ast_to_json(node.right_node)

        op = node.op_tok.value if node.op_tok.value else node.op_tok.type

        left_val = left["value"] if left and left["value"] else left["type"]
        right_val = right["value"] if right and right["value"] else right["type"]

        result["value"] = f"{left_val} {op} {right_val}"

        result["children"].append(left)
        result["children"].append(right)

        return result

    # ---------------------------
    # HANDLE IF NODE
    # ---------------------------

    if node_type == "IfNode":

        for condition, body, _ in node.cases:
            result["children"].append(ast_to_json(condition))
            result["children"].append(ast_to_json(body))

        if node.else_case:
            result["children"].append(ast_to_json(node.else_case[0]))

        return result

    # ---------------------------
    # HANDLE FOR NODE
    # ---------------------------

    if node_type == "ForNode":

        result["value"] = node.var_name_tok.value

        result["children"].append(ast_to_json(node.start_value_node))
        result["children"].append(ast_to_json(node.end_value_node))

        if node.step_value_node:
            result["children"].append(ast_to_json(node.step_value_node))

        result["children"].append(ast_to_json(node.body_node))

        return result

    # ---------------------------
    # GENERIC CHILD SERIALIZATION
    # ---------------------------

    for attr, value in vars(node).items():

        if attr in ("pos_start", "pos_end", "tok", "var_name_tok", "op_tok"):
            continue

        if hasattr(value, "__dict__"):
            child = ast_to_json(value)
            if child:
                result["children"].append(child)

        elif isinstance(value, list):
            for item in value:
                if hasattr(item, "__dict__"):
                    child = ast_to_json(item)
                    if child:
                        result["children"].append(child)

        elif isinstance(value, tuple):
            for item in value:
                if hasattr(item, "__dict__"):
                    child = ast_to_json(item)
                    if child:
                        result["children"].append(child)

    return result

