

from .strings_with_arrows import *

import string
import os
import math
import random
import time
import datetime

# ===============================
# EXECUTION TRACE (FOR VISUALIZER)
# ===============================

EXECUTION_TRACE = []

# ===============================
# EXECUTION SNAPSHOT
# ===============================

# ===============================
# EXECUTION SNAPSHOT
# ===============================

def snapshot(node, context):
    user_symbols = {}

    if context and context.symbol_table:
        for k, v in context.symbol_table.symbols.items():
            # hide built-ins and constants
            if k in ("null", "true", "false"):
                continue
            if k.startswith("Math_"):
                continue
            if str(v).startswith("<built-in function"):
                continue

            user_symbols[k] = str(v)

    EXECUTION_TRACE.append({
        "node": type(node).__name__,
        "line": node.pos_start.ln + 1 if node and node.pos_start else None,
        "scope": context.display_name if context else None,
        "symbols": user_symbols
    })

#######################################
# CONSTANTS
#######################################

DIGITS = '0123456789'
LETTERS = string.ascii_letters
LETTERS_DIGITS = LETTERS + DIGITS

#######################################
# ERRORS
#######################################

class Error:
  def __init__(self, pos_start, pos_end, error_name, details):
    self.pos_start = pos_start
    self.pos_end = pos_end
    self.error_name = error_name
    self.details = details
  
  def as_string(self):
    result  = f'{self.error_name}: {self.details}\n'
    result += f'File {self.pos_start.fn}, line {self.pos_start.ln + 1}'
    result += '\n\n' + string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)
    return result

class IllegalCharError(Error):
  def __init__(self, pos_start, pos_end, details):
    super().__init__(pos_start, pos_end, 'Illegal Character', details)

class ExpectedCharError(Error):
  def __init__(self, pos_start, pos_end, details):
    super().__init__(pos_start, pos_end, 'Expected Character', details)

class InvalidSyntaxError(Error):
  def __init__(self, pos_start, pos_end, details=''):
    super().__init__(pos_start, pos_end, 'Invalid Syntax', details)

class DictKeyError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, 'Dictionary Key Error', details)

class DictSyntaxError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end, 'Dictionary Syntax Error', details)

class RTError(Error):
  def __init__(self, pos_start, pos_end, details, context):
    super().__init__(pos_start, pos_end, 'Runtime Error', details)
    self.context = context

  def as_string(self):
    result  = self.generate_traceback()
    result += f'{self.error_name}: {self.details}'
    result += '\n\n' + string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)
    return result

  def generate_traceback(self):
    result = ''
    pos = self.pos_start
    ctx = self.context

    while ctx:
      result = f'  File {pos.fn}, line {str(pos.ln + 1)}, in {ctx.display_name}\n' + result
      pos = ctx.parent_entry_pos
      ctx = ctx.parent

    return 'Traceback (most recent call last):\n' + result

#######################################
# POSITION
#######################################

class Position:
  def __init__(self, idx, ln, col, fn, ftxt):
    self.idx = idx
    self.ln = ln
    self.col = col
    self.fn = fn
    self.ftxt = ftxt

  def advance(self, current_char=None):
    self.idx += 1
    self.col += 1

    if current_char == '\n':
      self.ln += 1
      self.col = 0

    return self

  def copy(self):
    return Position(self.idx, self.ln, self.col, self.fn, self.ftxt)

#######################################
# TOKENS
#######################################

TT_INT				= 'INT'
TT_FLOAT    	= 'FLOAT'
TT_STRING			= 'STRING'
TT_IDENTIFIER	= 'IDENTIFIER'
TT_KEYWORD		= 'KEYWORD'
TT_PLUS     	= 'PLUS'
TT_MINUS    	= 'MINUS'
TT_MUL      	= 'MUL'
TT_DIV      	= 'DIV'
TT_POW				= 'POW' 
TT_EQ					= 'EQ'
TT_LPAREN   	= 'LPAREN'
TT_RPAREN   	= 'RPAREN'
TT_LSQUARE    = 'LSQUARE'
TT_RSQUARE    = 'RSQUARE'
TT_EE					= 'EE'
TT_NE					= 'NE'
TT_LT					= 'LT'
TT_GT					= 'GT'
TT_LTE				= 'LTE'
TT_GTE				= 'GTE'
TT_COMMA			= 'COMMA'
TT_ARROW			= 'ARROW'
TT_NEWLINE		= 'NEWLINE'
TT_EOF				= 'EOF'
TT_LBRACE    = 'LBRACE'
TT_RBRACE    = 'RBRACE'
TT_COLON     = 'COLON'
TT_DOT       = 'DOT'

KEYWORDS = [
  'cup',               #variable
  'and',               #and operator
  'or',                #or operator
  'not',               #not operator
  'hot',               #if 
  'mild',              #else if
  'chilled',           #else
  'for_chai',          #for loop
  'to_chai',           #to keyword
  'STEP',              #step
  'while_chai',        #while loop
  'brew',              #function
  'then_chai',         #then keyword
  'sip',               #end statement
  'return_chai',       #return keyword
  'continue_chai',     #continue keyword
  'break_chai',        #break keyword
  'try_chai',          #try Keyword
  'catch_chai',        #catch keyword
]

class Token:
  def __init__(self, type_, value=None, pos_start=None, pos_end=None):
      self.type = type_
      self.value = value

      # Initialize position attributes 
      if pos_start:
          self.pos_start = pos_start.copy()
      else:
          self.pos_start = None  

      if pos_end:
          self.pos_end = pos_end.copy()
      else:
          if self.pos_start:
              self.pos_end = self.pos_start.copy()  # Default pos_end to be the same as pos_start if not provided
              self.pos_end.advance()  # Move end position one step forward
          else:
              self.pos_end = None  # Handle case if neither is provided

  def matches(self, type_, value):
      return self.type == type_ and self.value == value

  def __repr__(self):
      if self.value:
          return f'{self.type}: {self.value}'
      return f'{self.type}'


#######################################
# LEXER
#######################################

class Lexer:
  def __init__(self, fn, text):
    self.fn = fn
    self.text = text
    self.pos = Position(-1, 0, -1, fn, text)
    self.current_char = None
    self.brace_level = 0 
    self.advance()
  
  def advance(self):
    self.pos.advance(self.current_char)
    self.current_char = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None

  def make_tokens(self):
    tokens = []

    while self.current_char != None:
      if self.current_char in ' \t':
        self.advance()
      elif self.current_char in '\r\n':
        # Handle Windows \r\n properly
        if self.current_char == '\r':
            self.advance()
            continue

        if self.brace_level == 0:
            tokens.append(Token(TT_NEWLINE, pos_start=self.pos))
        self.advance()
      elif self.current_char == '#':
        self.skip_comment()
      elif self.current_char in ';\n':
        tokens.append(Token(TT_NEWLINE, pos_start=self.pos))
        self.advance()
      elif self.current_char in DIGITS:
        tokens.append(self.make_number())
      elif self.current_char in LETTERS:
        tokens.append(self.make_identifier())
      elif self.current_char == '"':
        tokens.append(self.make_string())
      elif self.current_char == '+':
        tokens.append(Token(TT_PLUS, pos_start=self.pos))
        self.advance()
      elif self.current_char == '-':
        tokens.append(self.make_minus_or_arrow())
      elif self.current_char == '*':
        tokens.append(Token(TT_MUL, pos_start=self.pos))
        self.advance()
      elif self.current_char == '/':
        tokens.append(Token(TT_DIV, pos_start=self.pos))
        self.advance()
      elif self.current_char == '^':
        tokens.append(Token(TT_POW, pos_start=self.pos))
        self.advance()
      elif self.current_char == '(':
        tokens.append(Token(TT_LPAREN, pos_start=self.pos))
        self.advance()
      elif self.current_char == ')':
        tokens.append(Token(TT_RPAREN, pos_start=self.pos))
        self.advance()
      elif self.current_char == '[':
        tokens.append(Token(TT_LSQUARE, pos_start=self.pos))
        self.advance()
      elif self.current_char == ',':
        tokens.append(Token(TT_COMMA))
        self.advance()

      elif self.current_char == ']':
        tokens.append(Token(TT_RSQUARE, pos_start=self.pos))
        self.advance()
      elif self.current_char == '!':
        token, error = self.make_not_equals()
        if error: return [], error
        tokens.append(token)
      elif self.current_char == '=':
        tokens.append(self.make_equals())
      elif self.current_char == '<':
        tokens.append(self.make_less_than())
      elif self.current_char == '>':
        tokens.append(self.make_greater_than())
      elif self.current_char == ',':
        tokens.append(Token(TT_COMMA, pos_start=self.pos))
        self.advance()
      # Inside Lexer.make_tokens()
      elif self.current_char == '.':
          tokens.append(Token(TT_DOT, pos_start=self.pos))
          self.advance()
          
      elif self.current_char == '{':
          tokens.append(Token(TT_LBRACE, pos_start=self.pos))
          self.brace_level += 1
          self.advance()
      elif self.current_char == '}':
          tokens.append(Token(TT_RBRACE, pos_start=self.pos))
          self.brace_level -= 1 
          self.advance()
      elif self.current_char == ':':
          tokens.append(Token(TT_COLON, pos_start=self.pos))
          self.advance()
      else:
        pos_start = self.pos.copy()
        char = self.current_char
        self.advance()
        return [], IllegalCharError(pos_start, self.pos, "'" + char + "'")

    tokens.append(Token(TT_EOF, pos_start=self.pos))
    return tokens, None

  def make_number(self):
    num_str = ''
    dot_count = 0
    pos_start = self.pos.copy()

    while self.current_char != None and self.current_char in DIGITS + '.':
      if self.current_char == '.':
        if dot_count == 1: break
        dot_count += 1
      num_str += self.current_char
      self.advance()

    if dot_count == 0:
      return Token(TT_INT, int(num_str), pos_start, self.pos)
    else:
      return Token(TT_FLOAT, float(num_str), pos_start, self.pos)

  def make_string(self):
    string = ''
    pos_start = self.pos.copy()
    escape_character = False
    self.advance()

    escape_characters = {
      'n': '\n',
      't': '\t'
    }

    while self.current_char != None and (self.current_char != '"' or escape_character):
      if escape_character:
        string += escape_characters.get(self.current_char, self.current_char)
      else:
        if self.current_char == '\\':
          escape_character = True
        else:
          string += self.current_char
      self.advance()
      escape_character = False
    
    self.advance()
    return Token(TT_STRING, string, pos_start, self.pos)

  def make_identifier(self):
    id_str = ''
    pos_start = self.pos.copy()

    # Only include letters, digits, and underscores
    while self.current_char != None and self.current_char in LETTERS_DIGITS + '_':
        id_str += self.current_char
        self.advance()

    tok_type = TT_KEYWORD if id_str in KEYWORDS else TT_IDENTIFIER
    return Token(tok_type, id_str, pos_start, self.pos)

  def make_minus_or_arrow(self):
    tok_type = TT_MINUS
    pos_start = self.pos.copy()
    self.advance()

    if self.current_char == '>':
      self.advance()
      tok_type = TT_ARROW

    return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

  def make_not_equals(self):
    pos_start = self.pos.copy()
    self.advance()

    if self.current_char == '=':
      self.advance()
      return Token(TT_NE, pos_start=pos_start, pos_end=self.pos), None

    self.advance()
    return None, ExpectedCharError(pos_start, self.pos, "'=' (after '!')")
  
  def make_equals(self):
    tok_type = TT_EQ
    pos_start = self.pos.copy()
    self.advance()

    if self.current_char == '=':
      self.advance()
      tok_type = TT_EE

    return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

  def make_less_than(self):
    tok_type = TT_LT
    pos_start = self.pos.copy()
    self.advance()

    if self.current_char == '=':
      self.advance()
      tok_type = TT_LTE

    return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

  
  def make_greater_than(self):
    tok_type = TT_GT
    pos_start = self.pos.copy()
    self.advance()

    if self.current_char == '=':
      self.advance()
      tok_type = TT_GTE

    return Token(tok_type, pos_start=pos_start, pos_end=self.pos)

  def skip_comment(self):
    self.advance()

    while self.current_char != '\n':
      self.advance()

    self.advance()

#######################################
# NODES
#######################################

class NumberNode:
  def __init__(self, tok):
    self.tok = tok

    self.pos_start = self.tok.pos_start
    self.pos_end = self.tok.pos_end

  def __repr__(self):
    return f'{self.tok}'

class StringNode:
  def __init__(self, tok):
    self.tok = tok

    self.pos_start = self.tok.pos_start
    self.pos_end = self.tok.pos_end

  def __repr__(self):
    return f'{self.tok}'

class ListNode:
  def __init__(self, element_nodes, pos_start, pos_end, index_nodes=None):
    self.element_nodes = element_nodes
    self.pos_start = pos_start
    self.pos_end = pos_end
    self.index_nodes = index_nodes if index_nodes is not None else []
  
class DictNode:
    def __init__(self, pairs, pos_start, pos_end):
        self.pairs = pairs  # List of (key_node, value_node) tuples
        self.pos_start = pos_start
        self.pos_end = pos_end
        
class TryCatchNode:
    def __init__(self, try_block, catch_block, pos_start, pos_end):
        self.try_block = try_block
        self.catch_block = catch_block
        self.pos_start = pos_start
        self.pos_end = pos_end
    
class IndexAccessNode:
    def __init__(self, list_node, index_node, pos_start, pos_end):
        self.list_node = list_node
        self.index_node = index_node
        self.pos_start = pos_start
        self.pos_end = pos_end


class VarAccessNode:
  def __init__(self, var_name_tok):
    self.var_name_tok = var_name_tok

    self.pos_start = self.var_name_tok.pos_start
    self.pos_end = self.var_name_tok.pos_end

class VarAssignNode:
  def __init__(self, var_name_tok, value_node):
    self.var_name_tok = var_name_tok
    self.value_node = value_node

    self.pos_start = self.var_name_tok.pos_start
    self.pos_end = self.value_node.pos_end

class BinOpNode:
  def __init__(self, left_node, op_tok, right_node):
    self.left_node = left_node
    self.op_tok = op_tok
    self.right_node = right_node

    self.pos_start = self.left_node.pos_start
    self.pos_end = self.right_node.pos_end

  def __repr__(self):
    return f'({self.left_node}, {self.op_tok}, {self.right_node})'

class UnaryOpNode:
  def __init__(self, op_tok, node):
    self.op_tok = op_tok
    self.node = node

    self.pos_start = self.op_tok.pos_start
    self.pos_end = node.pos_end

  def __repr__(self):
    return f'({self.op_tok}, {self.node})'

class IfNode:
  def __init__(self, cases, else_case):
    self.cases = cases
    self.else_case = else_case

    self.pos_start = self.cases[0][0].pos_start
    self.pos_end = (self.else_case or self.cases[len(self.cases) - 1])[0].pos_end

class ForNode:
  def __init__(self, var_name_tok, start_value_node, end_value_node, step_value_node, body_node, should_return_null):
    self.var_name_tok = var_name_tok
    self.start_value_node = start_value_node
    self.end_value_node = end_value_node
    self.step_value_node = step_value_node
    self.body_node = body_node
    self.should_return_null = should_return_null

    self.pos_start = self.var_name_tok.pos_start
    self.pos_end = self.body_node.pos_end

class WhileNode:
  def __init__(self, condition_node, body_node, should_return_null):
    self.condition_node = condition_node
    self.body_node = body_node
    self.should_return_null = should_return_null

    self.pos_start = self.condition_node.pos_start
    self.pos_end = self.body_node.pos_end

class FuncDefNode:
  def __init__(self, var_name_tok, arg_name_toks, body_node, should_auto_return):
    self.var_name_tok = var_name_tok
    self.arg_name_toks = arg_name_toks
    self.body_node = body_node
    self.should_auto_return = should_auto_return

    if self.var_name_tok:
      self.pos_start = self.var_name_tok.pos_start
    elif len(self.arg_name_toks) > 0:
      self.pos_start = self.arg_name_toks[0].pos_start
    else:
      self.pos_start = self.body_node.pos_start

    self.pos_end = self.body_node.pos_end

class CallNode:
  def __init__(self, node_to_call, arg_nodes):
    self.node_to_call = node_to_call
    self.arg_nodes = arg_nodes

    self.pos_start = self.node_to_call.pos_start

    if len(self.arg_nodes) > 0:
      self.pos_end = self.arg_nodes[len(self.arg_nodes) - 1].pos_end
    else:
      self.pos_end = self.node_to_call.pos_end

class ReturnNode:
  def __init__(self, node_to_return, pos_start, pos_end):
    self.node_to_return = node_to_return

    self.pos_start = pos_start
    self.pos_end = pos_end

class ContinueNode:
  def __init__(self, pos_start, pos_end):
    self.pos_start = pos_start
    self.pos_end = pos_end

class BreakNode:
  def __init__(self, pos_start, pos_end):
    self.pos_start = pos_start
    self.pos_end = pos_end

#######################################
# PARSE RESULT
#######################################

class ParseResult:
  def __init__(self):
    self.error = None
    self.node = None
    self.last_registered_advance_count = 0
    self.advance_count = 0
    self.to_reverse_count = 0

  def register_advancement(self):
    self.last_registered_advance_count = 1
    self.advance_count += 1

  def register(self, res):
    self.last_registered_advance_count = res.advance_count
    self.advance_count += res.advance_count
    if res.error: self.error = res.error
    return res.node

  def try_register(self, res):
    if res.error:
      self.to_reverse_count = res.advance_count
      return None
    return self.register(res)

  def success(self, node):
    self.node = node
    return self

  def failure(self, error):
    if not self.error or self.last_registered_advance_count == 0:
      self.error = error
    return self

#######################################
# PARSER
#######################################

class Parser:
  def __init__(self, tokens):
    self.tokens = tokens
    self.tok_idx = -1
    self.advance()

  def advance(self):
    self.tok_idx += 1
    self.update_current_tok()
    return self.current_tok

  def reverse(self, amount=1):
    self.tok_idx -= amount
    self.update_current_tok()
    return self.current_tok

  def update_current_tok(self):
    if self.tok_idx >= 0 and self.tok_idx < len(self.tokens):
      self.current_tok = self.tokens[self.tok_idx]

  def parse(self):
    res = self.statements()
    if not res.error and self.current_tok.type != TT_EOF:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        "Token cannot appear after previous tokens"
      ))
    return res

  ###################################

  def statements(self):
    res = ParseResult()
    statements = []
    pos_start = self.current_tok.pos_start.copy()

    while self.current_tok.type == TT_NEWLINE:
      res.register_advancement()
      self.advance()

    statement = res.register(self.statement())
    if res.error: return res
    statements.append(statement)

    more_statements = True

    while True:
      newline_count = 0
      while self.current_tok.type == TT_NEWLINE:
        res.register_advancement()
        self.advance()
        newline_count += 1
      if newline_count == 0:
        more_statements = False
      
      if not more_statements: break
      statement = res.try_register(self.statement())
      if not statement:
        self.reverse(res.to_reverse_count)
        more_statements = False
        continue
      statements.append(statement)

    return res.success(ListNode(
      statements,
      pos_start,
      self.current_tok.pos_end.copy()
    ))

  def statement(self):
    res = ParseResult()
    pos_start = self.current_tok.pos_start.copy()

    if self.current_tok.matches(TT_KEYWORD, 'return_chai'):
      res.register_advancement()
      self.advance()

      expr = res.try_register(self.expr())
      if not expr:
        self.reverse(res.to_reverse_count)
      return res.success(ReturnNode(expr, pos_start, self.current_tok.pos_start.copy()))
    
    if self.current_tok.matches(TT_KEYWORD, 'continue_chai'):
      res.register_advancement()
      self.advance()
      return res.success(ContinueNode(pos_start, self.current_tok.pos_start.copy()))
    
    if self.current_tok.matches(TT_KEYWORD, 'try_chai'):
            res.register_advancement()
            self.advance()
            
            if not self.current_tok.matches(TT_KEYWORD, 'then_chai'):
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected 'then_chai' after 'try_chai'"
                ))
            
            res.register_advancement()
            self.advance()
            
            try_block = res.register(self.statements())
            if res.error: return res
            
            if not self.current_tok.matches(TT_KEYWORD, 'catch_chai'):
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected 'catch_chai' after try block"
                ))
            
            res.register_advancement()
            self.advance()
            
            if not self.current_tok.matches(TT_KEYWORD, 'then_chai'):
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected 'then_chai' after 'catch_chai'"
                ))
            
            res.register_advancement()
            self.advance()
            
            catch_block = res.register(self.statements())
            if res.error: return res
            
            if not self.current_tok.matches(TT_KEYWORD, 'sip'):
                return res.failure(InvalidSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected 'sip' to close try-catch block"
                ))
            
            res.register_advancement()
            self.advance()
            
            return res.success(TryCatchNode(
                try_block,
                catch_block,
                self.current_tok.pos_start,
                self.current_tok.pos_end
            ))
    
    if self.current_tok.matches(TT_KEYWORD, 'break_chai'):
      res.register_advancement()
      self.advance()
      return res.success(BreakNode(pos_start, self.current_tok.pos_start.copy()))

    expr = res.register(self.expr())
    if res.error:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        "Expected 'return_chai', 'continue_chai', 'break_chai', 'cup', 'hot', 'for_chai', 'while_chai', 'brew', int, float, identifier, '+', '-', '(', '[' or 'not'"
      ))
    return res.success(expr)

  def expr(self):
    res = ParseResult()

    if self.current_tok.matches(TT_KEYWORD, 'cup'):
      res.register_advancement()
      self.advance()

      if self.current_tok.type != TT_IDENTIFIER:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          "Expected identifier"
        ))

      var_name = self.current_tok
      res.register_advancement()
      self.advance()

      if self.current_tok.type != TT_EQ:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          "Expected '='"
        ))

      res.register_advancement()
      self.advance()
      expr = res.register(self.expr())
      if res.error: return res
      return res.success(VarAssignNode(var_name, expr))

    node = res.register(self.bin_op(self.comp_expr, ((TT_KEYWORD, 'and'), (TT_KEYWORD, 'or'))))

    if res.error:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        "Expected 'cup', 'hot', 'for_chai', 'while_chai', 'brew', int, float, identifier, '+', '-', '(', '[' or 'not'"
      ))

    return res.success(node)

  def comp_expr(self):
    res = ParseResult()

    if self.current_tok.matches(TT_KEYWORD, 'not'):
      op_tok = self.current_tok
      res.register_advancement()
      self.advance()

      node = res.register(self.comp_expr())
      if res.error: return res
      return res.success(UnaryOpNode(op_tok, node))
    
    node = res.register(self.bin_op(self.arith_expr, (TT_EE, TT_NE, TT_LT, TT_GT, TT_LTE, TT_GTE)))
    
    if res.error:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        "Expected int, float, identifier, '+', '-', '(', '[', 'hot', 'for_chai', 'while_chai', 'brew' or 'not'"
      ))

    return res.success(node)

  def arith_expr(self):
    return self.bin_op(self.term, (TT_PLUS, TT_MINUS))

  def term(self):
    return self.bin_op(self.factor, (TT_MUL, TT_DIV))

  def factor(self):
    res = ParseResult()
    tok = self.current_tok

    if tok.type in (TT_PLUS, TT_MINUS):
      res.register_advancement()
      self.advance()
      factor = res.register(self.factor())
      if res.error: return res
      return res.success(UnaryOpNode(tok, factor))

    return self.power()

  def power(self):
    return self.bin_op(self.call, (TT_POW, ), self.factor)

  def call(self):
    res = ParseResult()
    atom = res.register(self.atom())
    if res.error: return res

    if self.current_tok.type == TT_LPAREN:
      res.register_advancement()
      self.advance()
      arg_nodes = []

      if self.current_tok.type == TT_RPAREN:
        res.register_advancement()
        self.advance()
      else:
        arg_nodes.append(res.register(self.expr()))
        if res.error:
          return res.failure(InvalidSyntaxError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            "Expected ')', 'cup', 'hot', 'for_chai', 'while_chai', 'brew', int, float, identifier, '+', '-', '(', '[' or 'not'"
          ))

        while self.current_tok.type == TT_COMMA:
          res.register_advancement()
          self.advance()

          arg_nodes.append(res.register(self.expr()))
          if res.error: return res

        if self.current_tok.type != TT_RPAREN:
          return res.failure(InvalidSyntaxError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            f"Expected ',' or ')'"
          ))

        res.register_advancement()
        self.advance()
      return res.success(CallNode(atom, arg_nodes))
    return res.success(atom)

  def atom(self):
    res = ParseResult()
    tok = self.current_tok

    if tok.type in (TT_INT, TT_FLOAT):
        res.register_advancement()
        self.advance()
        return res.success(NumberNode(tok))

    elif tok.type == TT_STRING:
        res.register_advancement()
        self.advance()
        return res.success(StringNode(tok))

    elif tok.type == TT_IDENTIFIER:
        res.register_advancement()
        self.advance()
        current_node = VarAccessNode(tok)  # Start with base identifier

        # Handle chained access (obj.field[0].sub)
        while True:
            # Index access with [ ]
            if self.current_tok.type == TT_LSQUARE:
                res.register_advancement()
                self.advance()

                index_node = res.register(self.expr())
                if res.error: return res

                if self.current_tok.type != TT_RSQUARE:
                    return res.failure(InvalidSyntaxError(
                        self.current_tok.pos_start, self.current_tok.pos_end,
                        "Expected ']'"
                    ))

                res.register_advancement()
                self.advance()
                current_node = IndexAccessNode(
                    current_node, 
                    index_node,
                    current_node.pos_start,
                    self.current_tok.pos_end
                )

            # Dot access (convert obj.field to obj["field"])
            elif self.current_tok.type == TT_DOT:
                res.register_advancement()
                self.advance()

                if self.current_tok.type != TT_IDENTIFIER:
                    return res.failure(InvalidSyntaxError(
                        self.current_tok.pos_start, self.current_tok.pos_end,
                        "Expected identifier after '.'"
                    ))

                # Create string key from identifier
                field_name = self.current_tok.value
                key_tok = Token(TT_STRING, field_name, self.current_tok.pos_start, self.current_tok.pos_end)
                key_node = StringNode(key_tok)

                res.register_advancement()
                self.advance()
                current_node = IndexAccessNode(
                    current_node,
                    key_node,
                    current_node.pos_start,
                    self.current_tok.pos_end
                )

            else:
                break  # No more chained accesses

        return res.success(current_node)

    elif tok.type == TT_LPAREN:
        res.register_advancement()
        self.advance()
        expr = res.register(self.expr())
        if res.error: return res
        if self.current_tok.type == TT_RPAREN:
            res.register_advancement()
            self.advance()
            return res.success(expr)
        else:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected ')'"
            ))

    elif tok.type == TT_LSQUARE:
        list_expr = res.register(self.list_expr())
        if res.error: return res
        return res.success(list_expr)

    elif tok.matches(TT_KEYWORD, 'hot'):
        if_expr = res.register(self.if_expr())
        if res.error: return res
        return res.success(if_expr)

    elif tok.matches(TT_KEYWORD, 'for_chai'):
        for_expr = res.register(self.for_expr())
        if res.error: return res
        return res.success(for_expr)

    elif tok.matches(TT_KEYWORD, 'while_chai'):
        while_expr = res.register(self.while_expr())
        if res.error: return res
        return res.success(while_expr)

    elif tok.matches(TT_KEYWORD, 'brew'):
        func_def = res.register(self.func_def())
        if res.error: return res
        return res.success(func_def)

    elif self.current_tok.type == TT_LBRACE:
        return self.dict_expr()

    return res.failure(InvalidSyntaxError(
        tok.pos_start, tok.pos_end,
        "Expected int, float, identifier, '+', '-', '(', '[', 'hot', 'for_chai', 'while_chai', 'brew'"
    ))



  def list_expr(self):
    res = ParseResult()
    element_nodes = []
    pos_start = self.current_tok.pos_start.copy()

    if self.current_tok.type != TT_LSQUARE:
        return res.failure(InvalidSyntaxError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            f"Expected '['"
        ))

    res.register_advancement()
    self.advance()

    if self.current_tok.type == TT_RSQUARE:
        # Empty list
        res.register_advancement()
        self.advance()
    else:
        # Parse elements in the list
        element_nodes.append(res.register(self.expr()))
        if res.error:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected ']', 'cup', 'hot', 'for_chai', 'while_chai', 'brew', int, float, identifier, '+', '-', '(', '[' or 'not'"
            ))

        while self.current_tok.type == TT_COMMA:
            res.register_advancement()
            self.advance()
            element_nodes.append(res.register(self.expr()))
            if res.error:
                return res

        if self.current_tok.type != TT_RSQUARE:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected ',' or ']'"
            ))

        res.register_advancement()
        self.advance()

    # Check if there’s an index expression following the list
    index_nodes = []
    while self.current_tok.type == TT_LSQUARE:
        # Start of index expression
        res.register_advancement()
        self.advance()

        # Parse the index
        index_expr = res.register(self.expr())
        if res.error:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                "Expected integer expression for index"
            ))
        index_nodes.append(index_expr)

        # Ensure there’s a closing bracket
        if self.current_tok.type != TT_RSQUARE:
            return res.failure(InvalidSyntaxError(
                self.current_tok.pos_start, self.current_tok.pos_end,
                f"Expected ']'"
            ))

        res.register_advancement()
        self.advance()

    return res.success(ListNode(
        element_nodes,
        pos_start,
        self.current_tok.pos_end.copy(),
        index_nodes  # Pass indices to ListNode
    ))
    
  def dict_expr(self):
    res = ParseResult()
    pairs = []
    pos_start = self.current_tok.pos_start.copy()

    res.register_advancement()
    self.advance()

    if self.current_tok.type == TT_RBRACE:
        res.register_advancement()
        self.advance()
    else:
        while True:
            # Parse key
            key = res.register(self.expr())
            if res.error:
                return res.failure(DictSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected key"
                ))

            # Check for colon after key
            if self.current_tok.type != TT_COLON:
                return res.failure(DictSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected ':' after key"
                ))

            res.register_advancement()
            self.advance()

            # Parse value
            value = res.register(self.expr())
            if res.error:
                return res.failure(DictSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected value after ':'"
                ))

            pairs.append((key, value))

            # Check for comma or closing brace
            if self.current_tok.type == TT_COMMA:
                res.register_advancement()
                self.advance()
            elif self.current_tok.type == TT_RBRACE:
                break
            else:
                return res.failure(DictSyntaxError(
                    self.current_tok.pos_start, self.current_tok.pos_end,
                    "Expected ',' or '}' after key-value pair"
                ))

        res.register_advancement()
        self.advance()

    return res.success(DictNode(pairs, pos_start, self.current_tok.pos_end))
  
  def if_expr(self):
    res = ParseResult()
    all_cases = res.register(self.if_expr_cases('hot'))
    if res.error: return res
    cases, else_case = all_cases
    return res.success(IfNode(cases, else_case))

  def if_expr_b(self):
    return self.if_expr_cases('mild')
    
  def if_expr_c(self):
    res = ParseResult()
    else_case = None

    if self.current_tok.matches(TT_KEYWORD, 'chilled'):
      res.register_advancement()
      self.advance()

      if self.current_tok.type == TT_NEWLINE:
        res.register_advancement()
        self.advance()

        statements = res.register(self.statements())
        if res.error: return res
        else_case = (statements, True)

        if self.current_tok.matches(TT_KEYWORD, 'sip'):
          res.register_advancement()
          self.advance()
        else:
          return res.failure(InvalidSyntaxError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            "Expected 'sip'"
          ))
      else:
        expr = res.register(self.statement())
        if res.error: return res
        else_case = (expr, False)

    return res.success(else_case)

  def if_expr_b_or_c(self):
    res = ParseResult()
    cases, else_case = [], None

    if self.current_tok.matches(TT_KEYWORD, 'mild'):
      all_cases = res.register(self.if_expr_b())
      if res.error: return res
      cases, else_case = all_cases
    else:
      else_case = res.register(self.if_expr_c())
      if res.error: return res
    
    return res.success((cases, else_case))

  def if_expr_cases(self, case_keyword):
    res = ParseResult()
    cases = []
    else_case = None

    if not self.current_tok.matches(TT_KEYWORD, case_keyword):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected '{case_keyword}'"
      ))

    res.register_advancement()
    self.advance()

    condition = res.register(self.expr())
    if res.error: return res

    if not self.current_tok.matches(TT_KEYWORD, 'then_chai'):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected 'then_chai'"
      ))

    res.register_advancement()
    self.advance()

    if self.current_tok.type == TT_NEWLINE:
      res.register_advancement()
      self.advance()

      statements = res.register(self.statements())
      if res.error: return res
      cases.append((condition, statements, True))

      if self.current_tok.matches(TT_KEYWORD, 'sip'):
        res.register_advancement()
        self.advance()
      else:
        all_cases = res.register(self.if_expr_b_or_c())
        if res.error: return res
        new_cases, else_case = all_cases
        cases.extend(new_cases)
    else:
      expr = res.register(self.statement())
      if res.error: return res
      cases.append((condition, expr, False))

      all_cases = res.register(self.if_expr_b_or_c())
      if res.error: return res
      new_cases, else_case = all_cases
      cases.extend(new_cases)

    return res.success((cases, else_case))

  def for_expr(self):
    res = ParseResult()

    if not self.current_tok.matches(TT_KEYWORD, 'for_chai'):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected 'for_chai'"
      ))

    res.register_advancement()
    self.advance()

    if self.current_tok.type != TT_IDENTIFIER:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected identifier"
      ))

    var_name = self.current_tok
    res.register_advancement()
    self.advance()

    if self.current_tok.type != TT_EQ:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected '='"
      ))
    
    res.register_advancement()
    self.advance()

    start_value = res.register(self.expr())
    if res.error: return res

    if not self.current_tok.matches(TT_KEYWORD, 'to_chai'):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected 'to_chai'"
      ))
    
    res.register_advancement()
    self.advance()

    end_value = res.register(self.expr())
    if res.error: return res

    if self.current_tok.matches(TT_KEYWORD, 'STEP'):
      res.register_advancement()
      self.advance()

      step_value = res.register(self.expr())
      if res.error: return res
    else:
      step_value = None

    if not self.current_tok.matches(TT_KEYWORD, 'then_chai'):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected 'then_chai'"
      ))

    res.register_advancement()
    self.advance()

    if self.current_tok.type == TT_NEWLINE:
      res.register_advancement()
      self.advance()

      body = res.register(self.statements())
      if res.error: return res

      if not self.current_tok.matches(TT_KEYWORD, 'sip'):
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          f"Expected 'sip'"
        ))

      res.register_advancement()
      self.advance()

      return res.success(ForNode(var_name, start_value, end_value, step_value, body, True))
    
    body = res.register(self.statement())
    if res.error: return res

    return res.success(ForNode(var_name, start_value, end_value, step_value, body, False))

  def while_expr(self):
    res = ParseResult()

    if not self.current_tok.matches(TT_KEYWORD, 'while_chai'):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected 'while_chai'"
      ))

    res.register_advancement()
    self.advance()

    condition = res.register(self.expr())
    if res.error: return res

    if not self.current_tok.matches(TT_KEYWORD, 'then_chai'):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected 'then_chai'"
      ))

    res.register_advancement()
    self.advance()

    if self.current_tok.type == TT_NEWLINE:
      res.register_advancement()
      self.advance()

      body = res.register(self.statements())
      if res.error: return res

      if not self.current_tok.matches(TT_KEYWORD, 'sip'):
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          f"Expected 'sip'"
        ))

      res.register_advancement()
      self.advance()

      return res.success(WhileNode(condition, body, True))
    
    body = res.register(self.statement())
    if res.error: return res

    return res.success(WhileNode(condition, body, False))

  def func_def(self):
    res = ParseResult()

    if not self.current_tok.matches(TT_KEYWORD, 'brew'):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected 'brew'"
      ))

    res.register_advancement()
    self.advance()

    if self.current_tok.type == TT_IDENTIFIER:
      var_name_tok = self.current_tok
      res.register_advancement()
      self.advance()
      if self.current_tok.type != TT_LPAREN:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          f"Expected '('"
        ))
    else:
      var_name_tok = None
      if self.current_tok.type != TT_LPAREN:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          f"Expected identifier or '('"
        ))
    
    res.register_advancement()
    self.advance()
    arg_name_toks = []

    if self.current_tok.type == TT_IDENTIFIER:
      arg_name_toks.append(self.current_tok)
      res.register_advancement()
      self.advance()
      
      while self.current_tok.type == TT_COMMA:
        res.register_advancement()
        self.advance()

        if self.current_tok.type != TT_IDENTIFIER:
          return res.failure(InvalidSyntaxError(
            self.current_tok.pos_start, self.current_tok.pos_end,
            f"Expected identifier"
          ))

        arg_name_toks.append(self.current_tok)
        res.register_advancement()
        self.advance()
      
      if self.current_tok.type != TT_RPAREN:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          f"Expected ',' or ')'"
        ))
    else:
      if self.current_tok.type != TT_RPAREN:
        return res.failure(InvalidSyntaxError(
          self.current_tok.pos_start, self.current_tok.pos_end,
          f"Expected identifier or ')'"
        ))

    res.register_advancement()
    self.advance()

    if self.current_tok.type == TT_ARROW:
      res.register_advancement()
      self.advance()

      body = res.register(self.expr())
      if res.error: return res

      return res.success(FuncDefNode(
        var_name_tok,
        arg_name_toks,
        body,
        True
      ))
    
    if self.current_tok.type != TT_NEWLINE:
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected '->' or NEWLINE"
      ))

    res.register_advancement()
    self.advance()

    body = res.register(self.statements())
    if res.error: return res

    if not self.current_tok.matches(TT_KEYWORD, 'sip'):
      return res.failure(InvalidSyntaxError(
        self.current_tok.pos_start, self.current_tok.pos_end,
        f"Expected 'sip'"
      ))

    res.register_advancement()
    self.advance()
    
    return res.success(FuncDefNode(
      var_name_tok,
      arg_name_toks,
      body,
      False
    ))

  ###################################

  def bin_op(self, func_a, ops, func_b=None):
    if func_b == None:
      func_b = func_a
    
    res = ParseResult()
    left = res.register(func_a())
    if res.error: return res

    while self.current_tok.type in ops or (self.current_tok.type, self.current_tok.value) in ops:
      op_tok = self.current_tok
      res.register_advancement()
      self.advance()
      right = res.register(func_b())
      if res.error: return res
      left = BinOpNode(left, op_tok, right)

    return res.success(left)

#######################################
# RUNTIME RESULT
#######################################

class RTResult:
  def __init__(self):
    self.reset()

  def reset(self):
    self.value = None
    self.error = None
    self.func_return_value = None
    self.loop_should_continue = False
    self.loop_should_break = False

  def register(self, res):
    self.error = res.error
    self.func_return_value = res.func_return_value
    self.loop_should_continue = res.loop_should_continue
    self.loop_should_break = res.loop_should_break
    return res.value

  def success(self, value):
    self.reset()
    self.value = value
    return self

  def success_return(self, value):
    self.reset()
    self.func_return_value = value
    return self
  
  def success_continue(self):
    self.reset()
    self.loop_should_continue = True
    return self

  def success_break(self):
    self.reset()
    self.loop_should_break = True
    return self

  def failure(self, error):
    self.reset()
    self.error = error
    return self

  def should_return(self):
    # Note: this will allow you to continue and break outside the current function
    return (
      self.error or
      self.func_return_value or
      self.loop_should_continue or
      self.loop_should_break
    )

#######################################
# VALUES
#######################################

class Value:
  def __init__(self):
    self.set_pos()
    self.set_context()

  def set_pos(self, pos_start=None, pos_end=None):
    self.pos_start = pos_start
    self.pos_end = pos_end
    return self

  def set_context(self, context=None):
    self.context = context
    return self

  def added_to(self, other):
    return None, self.illegal_operation(other)

  def subbed_by(self, other):
    return None, self.illegal_operation(other)

  def multed_by(self, other):
    return None, self.illegal_operation(other)

  def dived_by(self, other):
    return None, self.illegal_operation(other)

  def powed_by(self, other):
    return None, self.illegal_operation(other)

  def get_comparison_eq(self, other):
    return None, self.illegal_operation(other)

  def get_comparison_ne(self, other):
    return None, self.illegal_operation(other)

  def get_comparison_lt(self, other):
    return None, self.illegal_operation(other)

  def get_comparison_gt(self, other):
    return None, self.illegal_operation(other)

  def get_comparison_lte(self, other):
    return None, self.illegal_operation(other)

  def get_comparison_gte(self, other):
    return None, self.illegal_operation(other)

  def anded_by(self, other):
    return None, self.illegal_operation(other)

  def ored_by(self, other):
    return None, self.illegal_operation(other)

  def notted(self, other):
    return None, self.illegal_operation(other)

  def execute(self, args):
    return RTResult().failure(self.illegal_operation())

  def copy(self):
    raise Exception('No copy method defined')

  def is_true(self):
    return False

  def illegal_operation(self, other=None):
    if not other: other = self
    return RTError(
      self.pos_start, other.pos_end,
      'Illegal operation',
      self.context
    )

class Number(Value):
  def __init__(self, value):
        super().__init__()
        self.value = value

  def __eq__(self, other):
      return isinstance(other, Number) and self.value == other.value

  def __hash__(self):
      return hash(self.value)

  def added_to(self, other):
    if isinstance(other, Number):
      return Number(self.value + other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def subbed_by(self, other):
    if isinstance(other, Number):
      return Number(self.value - other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def multed_by(self, other):
    if isinstance(other, Number):
      return Number(self.value * other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def dived_by(self, other):
    if isinstance(other, Number):
      if other.value == 0:
        return None, RTError(
          other.pos_start, other.pos_end,
          'Division by zero',
          self.context
        )

      return Number(self.value / other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def powed_by(self, other):
    if isinstance(other, Number):
      return Number(self.value ** other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def get_comparison_eq(self, other):
    if isinstance(other, Number):
      return Number(str(self.value == other.value).lower()).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def get_comparison_ne(self, other):
    if isinstance(other, Number):
      return Number(str(self.value != other.value).lower()).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def get_comparison_lt(self, other):
    if isinstance(other, Number):
      return Number(str(self.value < other.value).lower()).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def get_comparison_gt(self, other):
    if isinstance(other, Number):
      return Number(str(self.value > other.value).lower()).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def get_comparison_lte(self, other):
    if isinstance(other, Number):
      return Number(str(self.value <= other.value).lower()).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def get_comparison_gte(self, other):
    if isinstance(other, Number):
      return Number(str(self.value >= other.value).lower()).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def anded_by(self, other):
    if isinstance(other, Number):
      return Number(str(self.value and other.value).lower()).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def ored_by(self, other):
    if isinstance(other, Number):
      return Number(str(self.value or other.value).lower()).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def notted(self):
    return Number("true" if self.value == "false" else "false").set_context(self.context), None

  def copy(self):
    copy = Number(self.value)
    copy.set_pos(self.pos_start, self.pos_end)
    copy.set_context(self.context)
    return copy

  def is_true(self):
    return self.value != "false"

  def __str__(self):
    return str(self.value)
  
  def __repr__(self):
    return str(self.value)

Number.null = Number("null")
Number.false = Number("false")
Number.true = Number("true")
Number.math_PI = Number(math.pi)
Number.math_E = Number(math.e)

class String(Value):
  def __init__(self, value):
        super().__init__()
        self.value = value

  def __eq__(self, other):
      return isinstance(other, String) and self.value == other.value

  def __hash__(self):
      return hash(self.value)

  def added_to(self, other):
    if isinstance(other, String):
      return String(self.value + other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def multed_by(self, other):
    if isinstance(other, Number):
      return String(self.value * other.value).set_context(self.context), None
    else:
      return None, Value.illegal_operation(self, other)

  def is_true(self):
    return len(self.value) > 0

  def copy(self):
    copy = String(self.value)
    copy.set_pos(self.pos_start, self.pos_end)
    copy.set_context(self.context)
    return copy

  def __str__(self):
    return self.value

  def __repr__(self):
    return f'"{self.value}"'

class List(Value):
  def __init__(self, elements):
    super().__init__()
    self.elements = elements

  def added_to(self, other):
    new_list = self.copy()
    new_list.elements.append(other)
    return new_list, None

  def subbed_by(self, other):
    if isinstance(other, Number):
      new_list = self.copy()
      try:
        new_list.elements.pop(other.value)
        return new_list, None
      except:
        return None, RTError(
          other.pos_start, other.pos_end,
          'Element at this index could not be removed from list because index is out of bounds',
          self.context
        )
    else:
      return None, Value.illegal_operation(self, other)

  def multed_by(self, other):
    if isinstance(other, List):
      new_list = self.copy()
      new_list.elements.extend(other.elements)
      return new_list, None
    else:
      return None, Value.illegal_operation(self, other)

  def dived_by(self, other):
    if isinstance(other, Number):
      try:
        return self.elements[other.value], None
      except:
        return None, RTError(
          other.pos_start, other.pos_end,
          'Element at this index could not be retrieved from list because index is out of bounds',
          self.context
        )
    else:
      return None, Value.illegal_operation(self, other)
  
  def copy(self):
    copy = List(self.elements)
    copy.set_pos(self.pos_start, self.pos_end)
    copy.set_context(self.context)
    return copy

  def __str__(self):
    return ", ".join([str(x) for x in self.elements])

  def __repr__(self):
    return f'[{", ".join([repr(x) for x in self.elements])}]'
  
  
class Dictionary(Value):
    def __init__(self, elements):
        super().__init__()
        # Store keys as Value objects (String/Number)
        self.elements = elements

    def get(self, key):
        # Extract the primitive value from the key (String/Number)
        if isinstance(key, (String, Number)):
            return self.elements.get(key.value, Number.null)  # Use key.value to access primitive key
        return Number.null

    # --------------------------------------------
    # Built-in Methods
    # --------------------------------------------
    def has(self, key):
        """Check if a key exists (key: String or Number)."""
        if not isinstance(key, (String, Number)):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Key must be a string or number",
                self.context
            ))
        key_value = key.value
        return Number.true if key_value in self.elements else Number.false

    def keys(self):
        """Return a list of keys as String/Number objects."""
        keys = []
        for k in self.elements.keys():
            keys.append(String(k) if isinstance(k, str) else Number(k))
        return List(keys)
    def values(self):
        """Return a list of values."""
        return List(list(self.elements.values()))

    def items(self):
        """Return a list of (key, value) tuples."""
        items = []
        for k, v in self.elements.items():
            key = String(k) if isinstance(k, str) else Number(k)
            items.append(List([key, v]))
        return List(items)

    def get(self, key, default=None):
        """Safe get with optional default."""
        return self.elements.get(key, default if default else Number.null)

    def pop(self, key, default=None):
        """Remove and return a key's value."""
        value = self.elements.pop(key, default if default else Number.null)
        return value

    def update(self, other_dict):
        """Merge another dictionary into this one."""
        if not isinstance(other_dict, Dictionary):
            return None, RTError(
                self.pos_start, self.pos_end,
                "Argument must be a dictionary",
                self.context
            )
        self.elements.update(other_dict.elements)
        return Number.null  # Return null to indicate success

    def clear(self):
        """Remove all key-value pairs."""
        self.elements.clear()
        return Number.null

    def copy(self):
        """Create a shallow copy."""
        return Dictionary(self.elements.copy())

    # --------------------------------------------
    # Utility Methods
    # --------------------------------------------
    def __repr__(self):
        pairs = []
        for k, v in self.elements.items():
            key_repr = repr(String(k)) if isinstance(k, str) else str(k)
            pairs.append(f"{key_repr}: {repr(v)}")
        return "{" + ", ".join(pairs) + "}"

    def __str__(self):
        return self.__repr__()
      


class BaseFunction(Value):
  def __init__(self, name):
    super().__init__()
    self.name = name or "<anonymous>"

  def generate_new_context(self):
    new_context = Context(self.name, self.context, self.pos_start)
    new_context.symbol_table = SymbolTable(new_context.parent.symbol_table)
    return new_context

  def check_args(self, arg_names, args):
    res = RTResult()

    if len(args) > len(arg_names):
      return res.failure(RTError(
        self.pos_start, self.pos_end,
        f"{len(args) - len(arg_names)} too many args passed into {self}",
        self.context
      ))
    
    if len(args) < len(arg_names):
      return res.failure(RTError(
        self.pos_start, self.pos_end,
        f"{len(arg_names) - len(args)} too few args passed into {self}",
        self.context
      ))

    return res.success(None)

  def populate_args(self, arg_names, args, exec_ctx):
    for i in range(len(args)):
      arg_name = arg_names[i]
      arg_value = args[i]
      arg_value.set_context(exec_ctx)
      exec_ctx.symbol_table.set(arg_name, arg_value)

  def check_and_populate_args(self, arg_names, args, exec_ctx):
    res = RTResult()
    res.register(self.check_args(arg_names, args))
    if res.should_return(): return res
    self.populate_args(arg_names, args, exec_ctx)
    return res.success(None)

class Function(BaseFunction):
  def __init__(self, name, body_node, arg_names, should_auto_return):
    super().__init__(name)
    self.body_node = body_node
    self.arg_names = arg_names
    self.should_auto_return = should_auto_return

  def execute(self, args):
    res = RTResult()
    interpreter = Interpreter()
    exec_ctx = self.generate_new_context()

    res.register(self.check_and_populate_args(self.arg_names, args, exec_ctx))
    if res.should_return(): return res

    value = res.register(interpreter.visit(self.body_node, exec_ctx))
    if res.should_return() and res.func_return_value == None: return res

    ret_value = (value if self.should_auto_return else None) or res.func_return_value or Number.null
    return res.success(ret_value)

  def copy(self):
    copy = Function(self.name, self.body_node, self.arg_names, self.should_auto_return)
    copy.set_context(self.context)
    copy.set_pos(self.pos_start, self.pos_end)
    return copy

  def __repr__(self):
    return f"<function {self.name}>"

class BuiltInFunction(BaseFunction):
  def __init__(self, name):
    super().__init__(name)

  def execute(self, args):
    res = RTResult()
    exec_ctx = self.generate_new_context()

    method_name = f'execute_{self.name}'
    method = getattr(self, method_name, self.no_visit_method)

    res.register(self.check_and_populate_args(method.arg_names, args, exec_ctx))
    if res.should_return(): return res

    return_value = res.register(method(exec_ctx))
    if res.should_return(): return res
    return res.success(return_value)
  
  def no_visit_method(self, node, context):
    raise Exception(f'No execute_{self.name} method defined')

  def copy(self):
    copy = BuiltInFunction(self.name)
    copy.set_context(self.context)
    copy.set_pos(self.pos_start, self.pos_end)
    return copy

  def __repr__(self):
    return f"<built-in function {self.name}>"

  #####################################

  def execute_print(self, exec_ctx):
    # Get the 'value' argument passed to print
    value = exec_ctx.symbol_table.get('value')

    # Check if value is of a compatible type and convert it to string if necessary
    if isinstance(value, (String, Number)):
        # Convert the value to a string representation
        output = str(value.value)
    elif isinstance(value, List):  # For lists, join elements with commas, for example
        output = ', '.join(str(item.value) for item in value.elements)
    else:
        output = str(value)  # Default conversion to string

    # Print the resulting output
    print(output)
    return RTResult().success(Number.null)

  # Set argument names for print function
  execute_print.arg_names = ['value']

  
  def execute_print_ret(self, exec_ctx):
    return RTResult().success(String(str(exec_ctx.symbol_table.get('value'))))
  execute_print_ret.arg_names = ['value']
  
  def execute_input(self, exec_ctx):
    text = input(str(exec_ctx.symbol_table.get('value'))) 
    return RTResult().success(String(text))
  execute_input.arg_names = ['value'] 

  def execute_input_int(self, exec_ctx):
    while True:
      text = input(str(exec_ctx.symbol_table.get('value')))
      try:
        number = int(text)
        break
      except ValueError:
        print(f"'{text}' must be an integer. Try again!")
    return RTResult().success(Number(number))
  execute_input_int.arg_names = ['value']

  def execute_clear(self, exec_ctx):
    os.system('cls' if os.name == 'nt' else 'cls') 
    return RTResult().success(Number.null)
  execute_clear.arg_names = []

  def execute_is_number(self, exec_ctx):
    is_number = isinstance(exec_ctx.symbol_table.get("value"), Number)
    return RTResult().success(Number.true if is_number else Number.false)
  execute_is_number.arg_names = ["value"]

  def execute_is_string(self, exec_ctx):
    is_number = isinstance(exec_ctx.symbol_table.get("value"), String)
    return RTResult().success(Number.true if is_number else Number.false)
  execute_is_string.arg_names = ["value"]

  def execute_is_list(self, exec_ctx):
    is_number = isinstance(exec_ctx.symbol_table.get("value"), List)
    return RTResult().success(Number.true if is_number else Number.false)
  execute_is_list.arg_names = ["value"]

  def execute_is_function(self, exec_ctx):
    is_number = isinstance(exec_ctx.symbol_table.get("value"), BaseFunction)
    return RTResult().success(Number.true if is_number else Number.false)
  execute_is_function.arg_names = ["value"]

  def execute_append(self, exec_ctx):
    list_ = exec_ctx.symbol_table.get("list")
    value = exec_ctx.symbol_table.get("value")

    if not isinstance(list_, List):
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        "First argument must be list",
        exec_ctx
      ))

    list_.elements.append(value)
    return RTResult().success(Number.null)
  execute_append.arg_names = ["list", "value"]

  def execute_pop(self, exec_ctx):
    list_ = exec_ctx.symbol_table.get("list")
    index = exec_ctx.symbol_table.get("index")

    if not isinstance(list_, List):
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        "First argument must be list",
        exec_ctx
      ))

    if not isinstance(index, Number):
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        "Second argument must be number",
        exec_ctx
      ))

    try:
      element = list_.elements.pop(index.value)
    except:
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        'Element at this index could not be removed from list because index is out of bounds',
        exec_ctx
      ))
    return RTResult().success(element)
  execute_pop.arg_names = ["list", "index"]

  def execute_extend(self, exec_ctx):
    listA = exec_ctx.symbol_table.get("listA")
    listB = exec_ctx.symbol_table.get("listB")

    if not isinstance(listA, List):
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        "First argument must be list",
        exec_ctx
      ))

    if not isinstance(listB, List):
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        "Second argument must be list",
        exec_ctx
      ))

    listA.elements.extend(listB.elements)
    return RTResult().success(Number.null)
  execute_extend.arg_names = ["listA", "listB"]
  
  def execute_sort(self, exec_ctx):
    value = exec_ctx.symbol_table.get("value")

    if isinstance(value, List):
        # Extract elements from the List instance
        elements = value.elements

        # Sort the elements (assuming they are all comparable)
        sorted_elements = sorted(elements, key=lambda x: x.value if isinstance(x, Number) else x)

        # Return a new List instance with sorted elements
        return RTResult().success(List(sorted_elements))
    else:
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Argument must be a list",
            exec_ctx
        ))

  execute_sort.arg_names = ["value"]
  
  def execute_reverse(self, exec_ctx):
    value = exec_ctx.symbol_table.get("value")

    if isinstance(value, List):
        reversed_elements = list(reversed(value.elements))
        return RTResult().success(List(reversed_elements))
    else:
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Argument must be a list",
            exec_ctx
        ))

  execute_reverse.arg_names = ["value"]

  def execute_range(self, exec_ctx):
    # Get the start, end, and step values from the arguments
    start = exec_ctx.symbol_table.get("start")
    end = exec_ctx.symbol_table.get("end")
    step = exec_ctx.symbol_table.get("step") if exec_ctx.symbol_table.get("step") else Number(1)

    # Validate that start, end, and step are numbers
    if not isinstance(start, Number) or not isinstance(end, Number) or not isinstance(step, Number):
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Arguments must be numbers",
            exec_ctx
        ))

    # Generate the range list
    range_list = []
    current = start.value

    # Ascending range
    if step.value > 0:
        while current < end.value:
            range_list.append(Number(current))
            current += step.value
    # Descending range
    elif step.value < 0:
        while current > end.value:
            range_list.append(Number(current))
            current += step.value
    else:
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Step cannot be zero",
            exec_ctx
        ))

    # Return the list as a ChaiScript list object
    return RTResult().success(List(range_list))

  # Set argument names
  execute_range.arg_names = ["start", "end", "step"]

  def execute_element_at(self, exec_ctx):
    # Get the list and index values from the arguments
    list_obj = exec_ctx.symbol_table.get("list")
    index = exec_ctx.symbol_table.get("index")

    # Validate that list_obj is a List
    if not isinstance(list_obj, List):
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "First argument must be a list",
            exec_ctx
        ))

    # Validate that index is a Number
    if not isinstance(index, Number):
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Second argument must be a number",
            exec_ctx
        ))

    # Try to access the element at the specified index
    try:
        element = list_obj.elements[index.value]
        return RTResult().success(element)
    except IndexError:
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            'Element at this index could not be retrieved because index is out of bounds',
            exec_ctx
        ))

  # Set argument names
  execute_element_at.arg_names = ["list", "index"]


  def execute_list_len(self, exec_ctx):
    list_ = exec_ctx.symbol_table.get("list")

    if not isinstance(list_, List):
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        "Argument must be list",
        exec_ctx
      ))

    return RTResult().success(Number(len(list_.elements)))
  execute_list_len.arg_names = ["list"]
  
  def execute_str_len(self, exec_ctx):
    string_value = exec_ctx.symbol_table.get("string")

    # Check if the argument is a String
    if not isinstance(string_value, String):
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Argument must be a string",
            exec_ctx
        ))

    # Return the length of the string
    return RTResult().success(Number(len(string_value.value)))

  # Define the argument names for the function
  execute_str_len.arg_names = ["string"]
  
  def execute_len(self, exec_ctx):
    value = exec_ctx.symbol_table.get("value")

    if isinstance(value, List):
        return RTResult().success(Number(len(value.elements)))
    elif isinstance(value, String):
        return RTResult().success(Number(len(value.value)))
    else:
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Argument must be a list or string",
            exec_ctx
        ))

  execute_len.arg_names = ["value"]
  
  def execute_to_lower(self, exec_ctx):
    value = exec_ctx.symbol_table.get("value")

    if not isinstance(value, String):
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Argument must be a string",
            exec_ctx
        ))

    # Convert the string to lowercase
    return RTResult().success(String(value.value.lower()))

  execute_to_lower.arg_names = ["value"]
  
  def execute_to_upper(self, exec_ctx):
    value = exec_ctx.symbol_table.get("value")

    if not isinstance(value, String):
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Argument must be a string",
            exec_ctx
        ))

    # Convert the string to lowercase
    return RTResult().success(String(value.value.upper()))

  execute_to_upper.arg_names = ["value"]
  
  def execute_char_at(self, exec_ctx):
    # Get the string and index from the execution context
    string_value = exec_ctx.symbol_table.get("string")
    index_value = exec_ctx.symbol_table.get("index")

    # Check if the first argument is a String
    if not isinstance(string_value, String):
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "First argument must be a string",
            exec_ctx
        ))

    # Check if the second argument is a Number
    if not isinstance(index_value, Number):
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Second argument must be an integer",
            exec_ctx
        ))

    # Convert the index to an integer and check bounds
    index = index_value.value
    if index < 0 or index >= len(string_value.value):
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Index out of bounds",
            exec_ctx
        ))

    # Return the character at the specified index as a string
    char = string_value.value[index]
    return RTResult().success(String(char))

  # Define the argument names for the function
  execute_char_at.arg_names = ["string", "index"]
  
  def execute_index_of(self, exec_ctx):
    # Get the string and character from the execution context
    string_value = exec_ctx.symbol_table.get("string")
    char_value = exec_ctx.symbol_table.get("char")

    # Check if the first argument is a String
    if not isinstance(string_value, String):
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "First argument must be a string",
            exec_ctx
        ))

    # Check if the second argument is a String and has only one character
    if not isinstance(char_value, String) or len(char_value.value) != 1:
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Second argument must be a single character",
            exec_ctx
        ))

    # Find the index of the character in the string
    index = string_value.value.find(char_value.value)

    if index == -1:
        index = "null"
    return RTResult().success(Number(index))

  # Define the argument names for the function
  execute_index_of.arg_names = ["string", "char"]
  
  
  def execute_slice(self, exec_ctx):
    # Get the string, start index, end index, and optional step from the execution context
    string_value = exec_ctx.symbol_table.get("string")
    start_index = exec_ctx.symbol_table.get("start")
    end_index = exec_ctx.symbol_table.get("end")
    step = exec_ctx.symbol_table.get("step")  # Default step is 1

    # Check if the first argument is a String
    if not isinstance(string_value, String):
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "First argument must be a string",
            exec_ctx
        ))

    # Check if start, end, and step are Numbers
    if not isinstance(start_index, Number) or not isinstance(end_index, Number) or not isinstance(step, Number):
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Start, end, and step must be numbers",
            exec_ctx
        ))

    # Slice the string using the start, end, and step
    sliced_string = string_value.value[start_index.value:end_index.value:step.value]
    
    # Return the sliced string
    return RTResult().success(String(sliced_string))

  # Define the argument names for the function
  execute_slice.arg_names = ["string", "start", "end", "step"]

  def execute_starts_with(self, exec_ctx):
    string = exec_ctx.symbol_table.get("string")
    substring = exec_ctx.symbol_table.get("substring")
    
    # Check if both arguments are instances of String
    if not isinstance(string, String) or not isinstance(substring, String):
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Arguments must be strings",
            exec_ctx
        ))

    # Check if the string starts with the substring
    result = string.value.startswith(substring.value)
    return RTResult().success(Number(Number.true if result else Number.false))

  # Set argument names 
  execute_starts_with.arg_names = ["string", "substring"]

  def execute_ends_with(self, exec_ctx):
    string = exec_ctx.symbol_table.get("string")
    substring = exec_ctx.symbol_table.get("substring")

    # Check if both arguments are instances of String
    if not isinstance(string, String) or not isinstance(substring, String):
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Arguments must be strings",
            exec_ctx
        ))

    # Check if the string ends with the substring
    result = string.value.endswith(substring.value)
    return RTResult().success(Number(Number.true if result else Number.false))

  # Set argument names
  execute_ends_with.arg_names = ["string", "substring"]

  def execute_trim(self, exec_ctx):
    string = exec_ctx.symbol_table.get("string")

    # Check if the argument is an instance of String
    if not isinstance(string, String):
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Argument must be a string",
            exec_ctx
        ))

    # Trim the string
    trimmed_value = string.value.strip()  # Removes leading and trailing whitespace
    return RTResult().success(String(trimmed_value))

  # Set argument names
  execute_trim.arg_names = ["string"]

  def execute_replace(self, exec_ctx):
    original_string = exec_ctx.symbol_table.get("original_string")
    to_replace = exec_ctx.symbol_table.get("to_replace")
    replacement = exec_ctx.symbol_table.get("replacement")

    # Check if the arguments are strings
    if not isinstance(original_string, String):
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "First argument must be a string",
            exec_ctx
        ))
    if not isinstance(to_replace, String):
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Second argument must be a string",
            exec_ctx
        ))
    if not isinstance(replacement, String):
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Third argument must be a string",
            exec_ctx
        ))

    # Perform the replacement
    replaced_string = original_string.value.replace(to_replace.value, replacement.value)
    return RTResult().success(String(replaced_string))

  # Set argument names
  execute_replace.arg_names = ["original_string", "to_replace", "replacement"]

  def execute_split(self, exec_ctx):
    original_string = exec_ctx.symbol_table.get("original_string")
    delimiter = exec_ctx.symbol_table.get("delimiter")

    # Check if the first argument is a string
    if not isinstance(original_string, String):
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "First argument must be a string",
            exec_ctx
        ))

    # Check if the second argument is a string
    if not isinstance(delimiter, String):
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Second argument must be a string",
            exec_ctx
        ))

    # Split the original string using the delimiter
    split_strings = original_string.value.split(delimiter.value)

    # Create a List object from the split strings
    split_list = List([String(s) for s in split_strings])
    
    return RTResult().success(split_list)

  # Set argument names 
  execute_split.arg_names = ["original_string", "delimiter"]

  def execute_join(self, exec_ctx):
    list_obj = exec_ctx.symbol_table.get("list_obj")
    delimiter = exec_ctx.symbol_table.get("delimiter")

    # Check if the first argument is a list
    if not isinstance(list_obj, List):
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "First argument must be a list",
            exec_ctx
        ))

    # Check if the second argument is a string
    if not isinstance(delimiter, String):
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Second argument must be a string",
            exec_ctx
        ))

    # Join the elements of the list using the delimiter
    joined_string = delimiter.value.join([str(element.value) for element in list_obj.elements])

    # Create a String object from the joined string
    result_string = String(joined_string)

    return RTResult().success(result_string)

  # C
  execute_join.arg_names = ["list_obj", "delimiter"]

  def execute_capitalize(self, exec_ctx):
    string = exec_ctx.symbol_table.get("string")

    # Check if the argument is an instance of String
    if not isinstance(string, String):
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Argument must be a string",
            exec_ctx
        ))

    # Capitalize the first letter of the string
    capitalized_value = string.value.capitalize()
    return RTResult().success(String(capitalized_value))

  #Set argument names
  execute_capitalize.arg_names = ["string"]
  
  def execute_title(self, exec_ctx):
    string = exec_ctx.symbol_table.get("string")

    # Check if the argument is an instance of String
    if not isinstance(string, String):
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Argument must be a string",
            exec_ctx
        ))

    # Title case the string
    title_value = string.value.title()
    return RTResult().success(String(title_value))

  #Set argument names
  execute_title.arg_names = ["string"]

  def execute_is_equals(self, exec_ctx):
    str1 = exec_ctx.symbol_table.get("str1")
    str2 = exec_ctx.symbol_table.get("str2")
    
    # Check if both are instances of String and compare their values
    if isinstance(str1, String) and isinstance(str2, String):
        return RTResult().success(Number.true if str1.value == str2.value else Number.false)
    else:
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Both arguments must be strings",
            exec_ctx
        ))

  execute_is_equals.arg_names = ["str1", "str2"]


  def execute_factorial(self, exec_ctx):
    # Get the number from the execution context
    number = exec_ctx.symbol_table.get("value")

    # Check if the argument is a Number and is non-negative
    if not isinstance(number, Number):
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Argument must be a number",
            exec_ctx
        ))
    if number.value < 0:
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Argument must be a non-negative integer",
            exec_ctx
        ))

    # Calculate factorial iteratively to avoid recursion depth issues
    result = 1
    for i in range(1, int(number.value) + 1):
        result *= i

    # Return the factorial result
    return RTResult().success(Number(result))

  # Define the argument name for the function
  execute_factorial.arg_names = ["value"]


  def execute_sqrt(self, exec_ctx):
    value = exec_ctx.symbol_table.get("value")

    # Check if the value is a number
    if not isinstance(value, Number):
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Argument must be a number",
            exec_ctx
        ))

    # Check if the value is non-negative
    if value.value < 0:
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Cannot compute the square root of a negative number",
            exec_ctx
        ))

    # Compute the square root
    result = math.sqrt(value.value)
    return RTResult().success(Number(result))

  execute_sqrt.arg_names = ["value"]

  def execute_floor(self, exec_ctx):
    value = exec_ctx.symbol_table.get("value")

    # Check if the value is a number
    if not isinstance(value, Number):
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Argument must be a number",
            exec_ctx
        ))

    # Compute the floor value
    result = math.floor(value.value)
    return RTResult().success(Number(result))

  execute_floor.arg_names = ["value"]
  
  def execute_ceil(self, exec_ctx):
    number = exec_ctx.symbol_table.get("number")

    # Check if the argument is an instance of Number
    if not isinstance(number, Number):
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Argument must be a number",
            exec_ctx
        ))

    # Calculate the ceiling value
    ceil_value = math.ceil(number.value)
    return RTResult().success(Number(ceil_value))

  #Set argument names
  execute_ceil.arg_names = ["number"]
  
  def execute_round(self, exec_ctx):
    number = exec_ctx.symbol_table.get("number")

    # Check if the argument is an instance of Number
    if not isinstance(number, Number):
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Argument must be a number",
            exec_ctx
        ))

    # Calculate the rounded value
    rounded_value = round(number.value)
    return RTResult().success(Number(rounded_value))

  #Set argument names
  execute_round.arg_names = ["number"]

  def execute_abs(self, exec_ctx):
    number = exec_ctx.symbol_table.get("number")

    # Check if the argument is an instance of Number
    if not isinstance(number, Number):
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Argument must be a number",
            exec_ctx
        ))

    # Calculate the absolute value
    abs_value = abs(number.value)
    return RTResult().success(Number(abs_value))

  #Set argument names
  execute_abs.arg_names = ["number"]


    
  def execute_randint(self, exec_ctx):
    min_value = exec_ctx.symbol_table.get("min")
    max_value = exec_ctx.symbol_table.get("max")

    # Check if both min and max are numbers
    if not isinstance(min_value, Number) or not isinstance(max_value, Number):
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Both arguments must be numbers",
            exec_ctx
        ))

    # Generate a random integer between min and max (inclusive)
    result = random.randint(min_value.value, max_value.value)
    return RTResult().success(Number(result))

  execute_randint.arg_names = ["min", "max"]
  
  import math

  def execute_sin(self, exec_ctx):
    value = exec_ctx.symbol_table.get("value")

    if isinstance(value, Number):  # Assuming 'Number' is your class for numeric types
        angle = value.value  # Angle is assumed to be in radians
        return RTResult().success(Number(math.sin(angle)))
    else:
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Argument must be a number",
            exec_ctx
        ))

  execute_sin.arg_names = ["value"]
  
  def execute_cos(self, exec_ctx):
    value = exec_ctx.symbol_table.get("value")

    if isinstance(value, Number):
        return RTResult().success(Number(math.cos(value.value)))
    else:
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Argument must be a number",
            exec_ctx
        ))

  execute_cos.arg_names = ["value"]

  def execute_tan(self, exec_ctx):
    value = exec_ctx.symbol_table.get("value")

    if isinstance(value, Number):
        # Check for a case where cosine is zero to avoid division by zero
        cos_value = math.cos(value.value)
        if cos_value == 0:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Tangent is undefined for this input (cosine is zero)",
                exec_ctx
            ))
        return RTResult().success(Number(math.tan(value.value)))
    else:
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Argument must be a number",
            exec_ctx
        ))

  execute_tan.arg_names = ["value"]

  def execute_sec(self, exec_ctx):
    value = exec_ctx.symbol_table.get("value")

    if isinstance(value, Number):
        cos_value = math.cos(value.value)
        if cos_value == 0:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Secant is undefined for this input (cosine is zero)",
                exec_ctx
            ))
        return RTResult().success(Number(1 / cos_value))
    else:
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Argument must be a number",
            exec_ctx
        ))

  execute_sec.arg_names = ["value"]

  def execute_cosec(self, exec_ctx):
    value = exec_ctx.symbol_table.get("value")

    if isinstance(value, Number):
        sin_value = math.sin(value.value)
        if sin_value == 0:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Cosecant is undefined for this input (sine is zero)",
                exec_ctx
            ))
        return RTResult().success(Number(1 / sin_value))
    else:
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Argument must be a number",
            exec_ctx
        ))

  execute_cosec.arg_names = ["value"]

  def execute_cot(self, exec_ctx):
    value = exec_ctx.symbol_table.get("value")

    if isinstance(value, Number):
        tan_value = math.tan(value.value)
        if tan_value == 0:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Cotangent is undefined for this input (tangent is zero)",
                exec_ctx
            ))
        return RTResult().success(Number(1 / tan_value))
    else:
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Argument must be a number",
            exec_ctx
        ))

  execute_cot.arg_names = ["value"]
  
  def execute_arcsin(self, exec_ctx):
    value = exec_ctx.symbol_table.get("value")

    if isinstance(value, Number):
        if value.value < -1 or value.value > 1:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Input must be in the range [-1, 1]",
                exec_ctx
            ))
        return RTResult().success(Number(math.asin(value.value)))
    else:
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Argument must be a number",
            exec_ctx
        ))

  execute_arcsin.arg_names = ["value"]
  
  def execute_arccos(self, exec_ctx):
    value = exec_ctx.symbol_table.get("value")

    if isinstance(value, Number):
        if value.value < -1 or value.value > 1:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Input must be in the range [-1, 1]",
                exec_ctx
            ))
        return RTResult().success(Number(math.acos(value.value)))
    else:
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Argument must be a number",
            exec_ctx
        ))

  execute_arccos.arg_names = ["value"]

  def execute_arctan(self, exec_ctx):
    value = exec_ctx.symbol_table.get("value")

    if isinstance(value, Number):
        return RTResult().success(Number(math.atan(value.value)))
    else:
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Argument must be a number",
            exec_ctx
        ))

  execute_arctan.arg_names = ["value"]

  def execute_arcsec(self, exec_ctx):
    value = exec_ctx.symbol_table.get("value")

    if isinstance(value, Number):
        if abs(value.value) < 1:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Input must be less than -1 or greater than 1",
                exec_ctx
            ))
        return RTResult().success(Number(math.acos(1 / value.value)))
    else:
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Argument must be a number",
            exec_ctx
        ))

  execute_arcsec.arg_names = ["value"]

  def execute_arccosec(self, exec_ctx):
    value = exec_ctx.symbol_table.get("value")

    if isinstance(value, Number):
        if abs(value.value) < 1:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Input must be less than -1 or greater than 1",
                exec_ctx
            ))
        return RTResult().success(Number(math.asin(1 / value.value)))
    else:
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Argument must be a number",
            exec_ctx
        ))

  execute_arccosec.arg_names = ["value"]
  
  def execute_arccot(self, exec_ctx):
    value = exec_ctx.symbol_table.get("value")

    if isinstance(value, Number):
        return RTResult().success(Number(math.atan(1 / value.value)))
    else:
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Argument must be a number",
            exec_ctx
        ))

  execute_arccot.arg_names = ["value"]

  def execute_sinh(self, exec_ctx):
    value = exec_ctx.symbol_table.get("value")

    if isinstance(value, Number):
        return RTResult().success(Number(math.sinh(value.value)))
    else:
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Argument must be a number",
            exec_ctx
        ))

  execute_sinh.arg_names = ["value"]
  
  def execute_cosh(self, exec_ctx):
    value = exec_ctx.symbol_table.get("value")

    if isinstance(value, Number):
        return RTResult().success(Number(math.cosh(value.value)))
    else:
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Argument must be a number",
            exec_ctx
        ))

  execute_cosh.arg_names = ["value"]

  def execute_tanh(self, exec_ctx):
    value = exec_ctx.symbol_table.get("value")

    if isinstance(value, Number):
        return RTResult().success(Number(math.tanh(value.value)))
    else:
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Argument must be a number",
            exec_ctx
        ))

  execute_tanh.arg_names = ["value"]
  
  def execute_sech(self, exec_ctx):
    value = exec_ctx.symbol_table.get("value")

    if isinstance(value, Number):
        cosh_value = math.cosh(value.value)
        if cosh_value == 0:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Input cannot be such that cosh(value) = 0",
                exec_ctx
            ))
        return RTResult().success(Number(1 / cosh_value))
    else:
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Argument must be a number",
            exec_ctx
        ))

  execute_sech.arg_names = ["value"]

  def execute_cosech(self, exec_ctx):
    value = exec_ctx.symbol_table.get("value")

    if isinstance(value, Number):
        sinh_value = math.sinh(value.value)
        if sinh_value == 0:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Input cannot be such that sinh(value) = 0",
                exec_ctx
            ))
        return RTResult().success(Number(1 / sinh_value))
    else:
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Argument must be a number",
            exec_ctx
        ))

  execute_cosech.arg_names = ["value"]
  
  def execute_coth(self, exec_ctx):
    value = exec_ctx.symbol_table.get("value")

    if isinstance(value, Number):
        sinh_value = math.sinh(value.value)
        if sinh_value == 0:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Input cannot be such that sinh(value) = 0",
                exec_ctx
            ))
        return RTResult().success(Number(sinh_value / sinh_value))
    else:
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Argument must be a number",
            exec_ctx
        ))

  execute_coth.arg_names = ["value"]

  def execute_arcsinh(self, exec_ctx):
        value = exec_ctx.symbol_table.get("value")

        if isinstance(value, Number):
            return RTResult().success(Number(math.asinh(value.value)))
        else:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Argument must be a number",
                exec_ctx
            ))

  execute_arcsinh.arg_names = ["value"]
  
  def execute_arccosh(self, exec_ctx):
      value = exec_ctx.symbol_table.get("value")

      if isinstance(value, Number):
          if value.value >= 1:
              return RTResult().success(Number(math.acosh(value.value)))
          else:
              return RTResult().failure(RTError(
                  self.pos_start, self.pos_end,
                  "Argument must be greater than or equal to 1",
                  exec_ctx
              ))
      else:
          return RTResult().failure(RTError(
              self.pos_start, self.pos_end,
              "Argument must be a number",
              exec_ctx
          ))

  execute_arccosh.arg_names = ["value"]

  def execute_arctanh(self, exec_ctx):
      value = exec_ctx.symbol_table.get("value")

      if isinstance(value, Number):
          if -1 < value.value < 1:
              return RTResult().success(Number(math.atanh(value.value)))
          else:
              return RTResult().failure(RTError(
                  self.pos_start, self.pos_end,
                  "Argument must be between -1 and 1",
                  exec_ctx
              ))
      else:
          return RTResult().failure(RTError(
              self.pos_start, self.pos_end,
              "Argument must be a number",
              exec_ctx
          ))
          
  execute_arctanh.arg_names = ["value"]

  def execute_arcsech(self, exec_ctx):
    value = exec_ctx.symbol_table.get("value")

    if not isinstance(value, Number):
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Argument must be a number",
            exec_ctx
        ))

    x = value.value
    
    # Check if the absolute value of the number is >= 1
    if abs(x) < 1:
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Argument must be greater than or equal to 1 or less than or equal to -1",
            exec_ctx
        ))

    # Calculate arcsech using the correct formula
    # Note: This calculation is valid because we've checked that |x| >= 1
    arcsech_value = math.log((1 + math.sqrt(x**2 - 1)) / x)

    return RTResult().success(Number(arcsech_value))

  execute_arcsech.arg_names = ["value"]

  
  def execute_arccosech(self, exec_ctx):
      value = exec_ctx.symbol_table.get("value")

      if isinstance(value, Number):
          if value.value != 0:
              return RTResult().success(Number(math.asinh(1/value.value)))
          else:
              return RTResult().failure(RTError(
                  self.pos_start, self.pos_end,
                  "Argument must not be zero",
                  exec_ctx
              ))
      else:
          return RTResult().failure(RTError(
              self.pos_start, self.pos_end,
              "Argument must be a number",
              exec_ctx
          ))

  execute_arccosech.arg_names = ["value"]
  
  def execute_arccoth(self, exec_ctx):
      value = exec_ctx.symbol_table.get("value")

      if isinstance(value, Number):
          if abs(value.value) > 1:
              return RTResult().success(Number(0.5 * math.log((value.value + 1)/(value.value - 1))))
          else:
              return RTResult().failure(RTError(
                  self.pos_start, self.pos_end,
                  "Argument must be greater than 1 or less than -1",
                  exec_ctx
              ))
      else:
          return RTResult().failure(RTError(
              self.pos_start, self.pos_end,
              "Argument must be a number",
              exec_ctx
          ))

  execute_arccoth.arg_names = ["value"]
  
  def execute_mod_div(self, exec_ctx):
        # Retrieve the arguments from the execution context
        num1 = exec_ctx.symbol_table.get("num1")
        num2 = exec_ctx.symbol_table.get("num2")

        # Check if both arguments are of the Number type
        if not isinstance(num1, Number) or not isinstance(num2, Number):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Both arguments must be numbers",
                exec_ctx
            ))

        # Calculate the remainder and return as a success result
        remainder = num1.value % num2.value
        return RTResult().success(Number(remainder))

  # Set argument names for the function
  execute_mod_div.arg_names = ["num1", "num2"]
  
  def execute_type(self, exec_ctx):
    value = exec_ctx.symbol_table.get("value")

    if value is None or value == Number.null: 
        return RTResult().success(String("null"))
    elif isinstance(value, Number):
        return RTResult().success(String("number"))
    elif isinstance(value, String):
        return RTResult().success(String("string"))
    elif isinstance(value, List):
        return RTResult().success(String("list"))
    elif isinstance(value, Function): 
        return RTResult().success(String("function"))
    else:
        return RTResult().success(String("unknown"))

  execute_type.arg_names = ["value"]
  
  def execute_int(self, exec_ctx):
    value = exec_ctx.symbol_table.get("value")  # Changed to 'value' for generality

    # Check if the argument is an instance of String or Number
    if not isinstance(value, (String, Number)):
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Argument must be a string or number",
            exec_ctx
        ))

    # Convert based on type
    if isinstance(value, String):
        # Try to convert the string to an integer
        try:
            int_value = int(value.value)  # Convert the string to an integer
            return RTResult().success(Number(int_value))
        except ValueError:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Invalid string for conversion to int",
                exec_ctx
            ))
    elif isinstance(value, Number):
        # If it's already a Number, return it as an integer
        return RTResult().success(Number(int(value.value)))  # Convert to integer

  #Set argument names
  execute_int.arg_names = ["value"]

  def execute_str(self, exec_ctx):
    value = exec_ctx.symbol_table.get("value")

    # Check if the argument is an instance of String, Number, or List
    if not isinstance(value, (String, Number, List)):
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Argument must be a string, number, or list",
            exec_ctx
        ))

    if isinstance(value, String):
        # If it's already a String, return it as is
        return RTResult().success(value.copy())
    elif isinstance(value, Number):
        # Convert the Number to a string
        str_value = str(value.value)
        return RTResult().success(String(str_value))
    elif isinstance(value, List):
        # Convert each element in the List to a string
        list_str = [str(element) for element in value.elements]
        # Join the string representations with commas
        joined_str = ", ".join(list_str)
        return RTResult().success(String(joined_str))

  #Set argument names
  execute_str.arg_names = ["value"]

  def execute_float(self, exec_ctx):
    value = exec_ctx.symbol_table.get("value")

    # Check if the argument is an instance of Number or String
    if isinstance(value, Number):
        return RTResult().success(Number(float(value.value)))
    elif isinstance(value, String):
        try:
            return RTResult().success(Number(float(value.value)))
        except ValueError:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Could not convert string to float",
                exec_ctx
            ))
    elif isinstance(value, List):
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Argument must not be a list",
            exec_ctx
        ))

    return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        "Argument must be a number or a string",
        exec_ctx
    ))

  #Set argument names
  execute_float.arg_names = ["value"]


  def execute_current_date_time(self, exec_ctx):
    # Get the current time
    now = datetime.datetime.now()
    # Format it as a string
    current_time_str = now.strftime("%d %B, %Y %I:%M:%S %p")
    return RTResult().success(String(current_time_str))

  execute_current_date_time.arg_names = []
  
  def execute_is_empty(self, exec_ctx):
    value = exec_ctx.symbol_table.get("value")

    # Check if the argument is a String, List, or Dict
    if isinstance(value, String):
        return RTResult().success(Number(Number.true if len(value.value) == 0 else Number.false))
    elif isinstance(value, List):
        return RTResult().success(Number(Number.true if len(value.elements) == 0 else Number.false))
    else:
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Argument must be a string or list",
            exec_ctx
        ))

  #Set argument names
  execute_is_empty.arg_names = ["value"]



  def execute_time_freeze(self, exec_ctx):
    value = exec_ctx.symbol_table.get("value")

    if isinstance(value, Number):
        duration = value.value
        if duration > 0:
            time.sleep(duration)  # Freeze time for the specified duration
            return RTResult().success(Number(duration))  # Return the duration after freeze
        else:
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Duration must be a positive number",
                exec_ctx
            ))
    else:
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Argument must be a number",
            exec_ctx
        ))

  execute_time_freeze.arg_names = ["value"]
  
  
  
  def execute_run(self, exec_ctx):
    fn = exec_ctx.symbol_table.get("fn")

    if not isinstance(fn, String):
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        "Second argument must be string",
        exec_ctx
      ))

    fn = fn.value

    try:
      with open(fn, "r") as f:
        script = f.read()
    except Exception as e:
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        f"Failed to load script \"{fn}\"\n" + str(e),
        exec_ctx
      ))

    _, error = run(fn, script)
    
    if error:
      return RTResult().failure(RTError(
        self.pos_start, self.pos_end,
        f"Failed to finish executing script \"{fn}\"\n" +
        error.as_string(),
        exec_ctx
      ))

    return RTResult().success(Number.null)
  execute_run.arg_names = ["fn"]
  
  def execute_has(self, exec_ctx):
        dict_obj = exec_ctx.symbol_table.get("dict")
        key = exec_ctx.symbol_table.get("key")

        if not isinstance(dict_obj, Dictionary):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "First argument must be a dictionary",
                exec_ctx
            ))

        if not isinstance(key, (String, Number)):
            return RTResult().failure(RTError(
                self.pos_start, self.pos_end,
                "Key must be a string or number",
                exec_ctx
            ))

        exists = key.value in dict_obj.elements
        return RTResult().success(Number.true if exists else Number.false)
  execute_has.arg_names = ["dict", "key"]

  def execute_keys(self, exec_ctx):
      dict_obj = exec_ctx.symbol_table.get("dict")
      
      if not isinstance(dict_obj, Dictionary):
          return RTResult().failure(RTError(
              self.pos_start, self.pos_end,
              "Argument must be a dictionary",
              exec_ctx
          ))

      keys = [String(k) if isinstance(k, str) else Number(k) 
              for k in dict_obj.elements.keys()]
      return RTResult().success(List(keys))
  execute_keys.arg_names = ["dict"]

  def execute_values(self, exec_ctx):
      dict_obj = exec_ctx.symbol_table.get("dict")
      
      if not isinstance(dict_obj, Dictionary):
          return RTResult().failure(RTError(
              self.pos_start, self.pos_end,
              "Argument must be a dictionary",
              exec_ctx
          ))

      return RTResult().success(List(list(dict_obj.elements.values())))
  execute_values.arg_names = ["dict"]

  def execute_get(self, exec_ctx):
    dict_obj = exec_ctx.symbol_table.get("dict")
    key = exec_ctx.symbol_table.get("key")

    # Validate dictionary and key types
    if not isinstance(dict_obj, Dictionary):
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "First argument must be a dictionary",
            exec_ctx
        ))

    if not isinstance(key, (String, Number)):
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Key must be a string or number",
            exec_ctx
        ))

    # Extract the primitive value from the key (String/Number)
    key_primitive = key.value

    # Retrieve value using the primitive key
    value = dict_obj.elements.get(key_primitive, Number.null)
    return RTResult().success(value)

  execute_get.arg_names = ["dict", "key"]

  def execute_pop(self, exec_ctx):
      dict_obj = exec_ctx.symbol_table.get("dict")
      key = exec_ctx.symbol_table.get("key")
      default = exec_ctx.symbol_table.get("default")

      if not isinstance(dict_obj, Dictionary):
          return RTResult().failure(RTError(
              self.pos_start, self.pos_end,
              "First argument must be a dictionary",
              exec_ctx
          ))

      value = dict_obj.elements.pop(key.value, default if default else Number.null)
      return RTResult().success(value)
  execute_pop.arg_names = ["dict", "key", "default"]

  def execute_update(self, exec_ctx):
    dict1 = exec_ctx.symbol_table.get("dict1")
    dict2 = exec_ctx.symbol_table.get("dict2")

    if not isinstance(dict1, Dictionary) or not isinstance(dict2, Dictionary):
        return RTResult().failure(RTError(
            self.pos_start, self.pos_end,
            "Both arguments must be dictionaries",
            exec_ctx
        ))

    dict1.elements.update(dict2.elements)
    return RTResult().success(dict1)  # Return the updated dict1

  execute_update.arg_names = ["dict1", "dict2"]

  def execute_clear(self, exec_ctx):
      dict_obj = exec_ctx.symbol_table.get("dict")
      
      if not isinstance(dict_obj, Dictionary):
          return RTResult().failure(RTError(
              self.pos_start, self.pos_end,
              "Argument must be a dictionary",
              exec_ctx
          ))

      dict_obj.elements.clear()
      return RTResult().success(Number.null)
  execute_clear.arg_names = ["dict"]

  def execute_copy(self, exec_ctx):
      dict_obj = exec_ctx.symbol_table.get("dict")
      
      if not isinstance(dict_obj, Dictionary):
          return RTResult().failure(RTError(
              self.pos_start, self.pos_end,
              "Argument must be a dictionary",
              exec_ctx
          ))

      return RTResult().success(Dictionary(dict_obj.elements.copy()))
  execute_copy.arg_names = ["dict"]



BuiltInFunction.print       = BuiltInFunction("print")
BuiltInFunction.print_ret   = BuiltInFunction("print_ret")
BuiltInFunction.input       = BuiltInFunction("input")
BuiltInFunction.input_int   = BuiltInFunction("input_int")
BuiltInFunction.clear       = BuiltInFunction("clear")
BuiltInFunction.is_number   = BuiltInFunction("is_number")
BuiltInFunction.is_string   = BuiltInFunction("is_string")
BuiltInFunction.is_list     = BuiltInFunction("is_list")
BuiltInFunction.is_function = BuiltInFunction("is_function")
BuiltInFunction.append      = BuiltInFunction("append")
BuiltInFunction.pop         = BuiltInFunction("pop")
BuiltInFunction.extend      = BuiltInFunction("extend")
BuiltInFunction.list_len		= BuiltInFunction("list_len")
BuiltInFunction.str_len		  = BuiltInFunction("str_len")
BuiltInFunction.len		      = BuiltInFunction("len")
BuiltInFunction.run					= BuiltInFunction("run")
BuiltInFunction.to_lower    = BuiltInFunction("to_lower")
BuiltInFunction.to_upper    = BuiltInFunction("to_upper")
BuiltInFunction.char_at     = BuiltInFunction("char_at")
BuiltInFunction.index_of    = BuiltInFunction("index_of")
BuiltInFunction.slice       = BuiltInFunction("slice")
BuiltInFunction.factorial   = BuiltInFunction("factorial")
BuiltInFunction.sqrt        = BuiltInFunction("sqrt")
BuiltInFunction.floor       = BuiltInFunction("floor")
BuiltInFunction.randint       = BuiltInFunction("randint")
BuiltInFunction.printf      = BuiltInFunction("printf")







#######################################
# CONTEXT
#######################################

class Context:
  def __init__(self, display_name, parent=None, parent_entry_pos=None):
    self.display_name = display_name
    self.parent = parent
    self.parent_entry_pos = parent_entry_pos
    self.symbol_table = None
    self.symbol_table = SymbolTable(parent.symbol_table if parent else None)

#######################################
# SYMBOL TABLE
#######################################

class SymbolTable:
  def __init__(self, parent=None):
    self.symbols = {}
    self.parent = parent

  def get(self, name):
    value = self.symbols.get(name, None)
    if value == None and self.parent:
      return self.parent.get(name)
    return value

  def set(self, name, value):
    self.symbols[name] = value

  def remove(self, name):
    del self.symbols[name]

#######################################
# INTERPRETER
#######################################

class Interpreter:
  def visit(self, node, context):
    method_name = f'visit_{type(node).__name__}'
    method = getattr(self, method_name, self.no_visit_method)

    node_name = type(node).__name__

    # 🔹 Snapshot control-flow nodes BEFORE execution
    if node_name in ("IfNode", "ForNode", "WhileNode"):
        snapshot(node, context)

    result = method(node, context)

    # 🔹 Snapshot other meaningful nodes AFTER execution
    if node_name in (
        "VarAssignNode",
        "FuncDefNode",
        "CallNode",
        "ReturnNode"
    ):
        snapshot(node, context)

    return result

  def no_visit_method(self, node, context):
    raise Exception(f'No visit_{type(node).__name__} method defined')

  ###################################

  def visit_NumberNode(self, node, context):
    return RTResult().success(
      Number(node.tok.value).set_context(context).set_pos(node.pos_start, node.pos_end)
    )

  def visit_StringNode(self, node, context):
    return RTResult().success(
      String(node.tok.value).set_context(context).set_pos(node.pos_start, node.pos_end)
    )

  def visit_ListNode(self, node, context):
    res = RTResult()
    elements = []

    for element_node in node.element_nodes:
      elements.append(res.register(self.visit(element_node, context)))
      if res.should_return(): return res

    return res.success(
      List(elements).set_context(context).set_pos(node.pos_start, node.pos_end)
    )

  def visit_VarAccessNode(self, node, context):
    res = RTResult()
    var_name = node.var_name_tok.value
    value = context.symbol_table.get(var_name)

    if not value:
      return res.failure(RTError(
        node.pos_start, node.pos_end,
        f"'{var_name}' is not defined",
        context
      ))

    value = value.copy().set_pos(node.pos_start, node.pos_end).set_context(context)
    return res.success(value)

  def visit_VarAssignNode(self, node, context):
    res = RTResult()
    var_name = node.var_name_tok.value
    value = res.register(self.visit(node.value_node, context))
    if res.should_return(): return res

    context.symbol_table.set(var_name, value)
    return res.success(value)

  def visit_BinOpNode(self, node, context):
    res = RTResult()
    left = res.register(self.visit(node.left_node, context))
    if res.should_return(): return res
    right = res.register(self.visit(node.right_node, context))
    if res.should_return(): return res

    result = None  # Initialize result
    error = None   # Initialize error

    if node.op_tok.type == TT_PLUS:
        # Check if one of the operands is a string
        if isinstance(left, String) or isinstance(right, String):
            # Convert both operands to strings and concatenate
            left_value = str(left.value) if isinstance(left, Number) else left.value
            right_value = str(right.value) if isinstance(right, Number) else right.value
            result = String(left_value + right_value)
        else:
            result, error = left.added_to(right)
    elif node.op_tok.type == TT_MINUS:
        result, error = left.subbed_by(right)
    elif node.op_tok.type == TT_MUL:
        result, error = left.multed_by(right)
    elif node.op_tok.type == TT_DIV:
        result, error = left.dived_by(right)
    elif node.op_tok.type == TT_POW:
        result, error = left.powed_by(right)
    elif node.op_tok.type == TT_EE:
        result, error = left.get_comparison_eq(right)
    elif node.op_tok.type == TT_NE:
        result, error = left.get_comparison_ne(right)
    elif node.op_tok.type == TT_LT:
        result, error = left.get_comparison_lt(right)
    elif node.op_tok.type == TT_GT:
        result, error = left.get_comparison_gt(right)
    elif node.op_tok.type == TT_LTE:
        result, error = left.get_comparison_lte(right)
    elif node.op_tok.type == TT_GTE:
        result, error = left.get_comparison_gte(right)
    elif node.op_tok.matches(TT_KEYWORD, 'and'):
        result, error = left.anded_by(right)
    elif node.op_tok.matches(TT_KEYWORD, 'or'):
        result, error = left.ored_by(right)

    if error:
        return res.failure(error)
    else:
        return res.success(result.set_pos(node.pos_start, node.pos_end))


  def visit_UnaryOpNode(self, node, context):
    res = RTResult()
    number = res.register(self.visit(node.node, context))
    if res.should_return(): return res

    error = None

    if node.op_tok.type == TT_MINUS:
      number, error = number.multed_by(Number(-1))
    elif node.op_tok.matches(TT_KEYWORD, 'not'):
      number, error = number.notted()

    if error:
      return res.failure(error)
    else:
      return res.success(number.set_pos(node.pos_start, node.pos_end))

  def visit_IfNode(self, node, context):
    res = RTResult()

    for condition, expr, should_return_null in node.cases:
      condition_value = res.register(self.visit(condition, context))
      if res.should_return(): return res

      if condition_value.is_true():
        expr_value = res.register(self.visit(expr, context))
        if res.should_return(): return res
        return res.success(Number.null if should_return_null else expr_value)

    if node.else_case:
      expr, should_return_null = node.else_case
      expr_value = res.register(self.visit(expr, context))
      if res.should_return(): return res
      return res.success(Number.null if should_return_null else expr_value)

    return res.success(Number.null)

  def visit_ForNode(self, node, context):
    res = RTResult()
    elements = []

    start_value = res.register(self.visit(node.start_value_node, context))
    if res.should_return(): return res

    end_value = res.register(self.visit(node.end_value_node, context))
    if res.should_return(): return res

    if node.step_value_node:
      step_value = res.register(self.visit(node.step_value_node, context))
      if res.should_return(): return res
    else:
      step_value = Number(1)

    i = start_value.value

    if step_value.value >= 0:
      condition = lambda: i < end_value.value
    else:
      condition = lambda: i > end_value.value
    
    while condition():
      context.symbol_table.set(node.var_name_tok.value, Number(i))
      i += step_value.value

      value = res.register(self.visit(node.body_node, context))
      if res.should_return() and res.loop_should_continue == False and res.loop_should_break == False: return res
      
      if res.loop_should_continue:
        continue
      
      if res.loop_should_break:
        break

      elements.append(value)

    return res.success(
      Number.null if node.should_return_null else
      List(elements).set_context(context).set_pos(node.pos_start, node.pos_end)
    )

  def visit_WhileNode(self, node, context):
    res = RTResult()
    elements = []

    while True:
      condition = res.register(self.visit(node.condition_node, context))
      if res.should_return(): return res

      if not condition.is_true():
        break

      value = res.register(self.visit(node.body_node, context))
      if res.should_return() and res.loop_should_continue == False and res.loop_should_break == False: return res

      if res.loop_should_continue:
        continue
      
      if res.loop_should_break:
        break

      elements.append(value)

    return res.success(
      Number.null if node.should_return_null else
      List(elements).set_context(context).set_pos(node.pos_start, node.pos_end)
    )

  def visit_FuncDefNode(self, node, context):
    res = RTResult()

    func_name = node.var_name_tok.value if node.var_name_tok else None
    body_node = node.body_node
    arg_names = [arg_name.value for arg_name in node.arg_name_toks]
    func_value = Function(func_name, body_node, arg_names, node.should_auto_return).set_context(context).set_pos(node.pos_start, node.pos_end)
    
    if node.var_name_tok:
      context.symbol_table.set(func_name, func_value)

    return res.success(func_value)

  def visit_CallNode(self, node, context):
        res = RTResult()
        args = []

        # Resolve the object (e.g., `person` in `person.keys()`)
        obj = res.register(self.visit(node.node_to_call, context))
        if res.error:
            return res

        # Resolve arguments
        for arg_node in node.arg_nodes:
            args.append(res.register(self.visit(arg_node, context)))
            if res.error:
                return res

        # Handle dictionary methods
        if isinstance(obj, Dictionary):
            method_name = node.method_name_tok.value  # e.g., "keys"
            method = getattr(obj, method_name, None)
            if method:
                # Call the method and return its result
                result = res.register(method(*args))
                return res.success(result)
            else:
                return res.failure(RTError(
                    node.pos_start, node.pos_end,
                    f"Dictionary has no method '{method_name}'",
                    context
                ))

        # Handle other function calls
        if not isinstance(obj, BaseFunction):
            return res.failure(RTError(
                node.pos_start, node.pos_end,
                "Expected function or method",
                context
            ))

        return_value = res.register(obj.execute(args))
        return res.success(return_value)

  def visit_ReturnNode(self, node, context):
    res = RTResult()

    if node.node_to_return:
      value = res.register(self.visit(node.node_to_return, context))
      if res.should_return(): return res
    else:
      value = Number.null
    
    return res.success_return(value)

  def visit_ContinueNode(self, node, context):
    return RTResult().success_continue()

  def visit_BreakNode(self, node, context):
    return RTResult().success_break()
  
  def visit_DictNode(self, node, context):
        res = RTResult()
        elements = {}
        seen_keys = set()

        for key_node, value_node in node.pairs:
            # Evaluate the key
            key = res.register(self.visit(key_node, context))
            if res.error:
                return res

            # Validate key type
            if not isinstance(key, (String, Number)):
                return res.failure(DictKeyError(
                    key_node.pos_start, key_node.pos_end,
                    "Dictionary keys must be strings or numbers"
                ))

            key_value = key.value

            # Check for duplicate keys
            if key_value in seen_keys:
                return res.failure(DictKeyError(
                    key_node.pos_start, key_node.pos_end,
                    f"Duplicate key '{key_value}'"
                ))
            seen_keys.add(key_value)

            # Evaluate the value
            value = res.register(self.visit(value_node, context))
            if res.error:
                return res

            elements[key_value] = value

        return res.success(Dictionary(elements))
      
  def visit_TryCatchNode(self, node, context):
    res = RTResult()
    
    # Execute try block
    try_result = res.register(self.visit(node.try_block, context))
    if res.error:
        error = res.error
        res.error = None  # Clear error to proceed to catch
        
        # Create error dictionary with primitive string keys
        error_dict = Dictionary({
            "message": String(error.details),
            "start_line": Number(error.pos_start.ln + 1),
            "end_line": Number(error.pos_end.ln + 1)
        })
        
        # Initialize catch context and set 'error' variable
        catch_context = Context('catch', context, node.pos_start)
        catch_context.symbol_table = SymbolTable(context.symbol_table)
        catch_context.symbol_table.set("error", error_dict)
        
        # Execute catch block
        catch_result = res.register(self.visit(node.catch_block, catch_context))
        if res.error: return res
        
        return res.success(catch_result)
    
    return res.success(try_result)
      
      
  def visit_IndexAccessNode(self, node, context):
    res = RTResult()
    
    # Evaluate the list/dictionary and index
    container = res.register(self.visit(node.list_node, context))
    if res.error: return res
    
    index = res.register(self.visit(node.index_node, context))
    if res.error: return res

    # Dictionary Access
    if isinstance(container, Dictionary):
        # Key must be a String or Number
        if not isinstance(index, (String, Number)):
            return res.failure(RTError(
                node.index_node.pos_start, node.index_node.pos_end,
                "Dictionary key must be string or number",
                context
            ))
        
        value = container.get(index)
        return res.success(value)
    
    # Evaluate the list/dictionary and index
    list_or_dict = res.register(self.visit(node.list_node, context))
    if res.error: return res
    
    index = res.register(self.visit(node.index_node, context))
    if res.error: return res

    # Case 1: List Indexing
    if isinstance(list_or_dict, List):
        # Validate index is a number
        if not isinstance(index, Number):
            return res.failure(RTError(
                node.index_node.pos_start, node.index_node.pos_end,
                "List index must be a number",
                context
            ))

        # Convert to integer index
        try:
            index = int(index.value)
        except:
            return res.failure(RTError(
                node.index_node.pos_start, node.index_node.pos_end,
                "List index must be an integer",
                context
            ))

        # Check bounds
        if index < 0 or index >= len(list_or_dict.elements):
            return res.failure(RTError(
                node.pos_start, node.pos_end,
                f"Index {index} out of range for list",
                context
            ))

        # Return element
        return res.success(list_or_dict.elements[index])

    # Case 2: Dictionary Indexing
    elif isinstance(list_or_dict, Dictionary):
        # Validate key is string/number
        if not isinstance(index, (String, Number)):
            return res.failure(RTError(
                node.index_node.pos_start, node.index_node.pos_end,
                "Dictionary key must be a string or number",
                context
            ))

        key = index.value  # Extract primitive value

        # Check existence
        if key not in list_or_dict.elements:
            return res.failure(RTError(
                node.pos_start, node.pos_end,
                f"Key '{key}' not found in dictionary",
                context
            ))

        return res.success(list_or_dict.elements[key])

    # Invalid type
    else:
        return res.failure(RTError(
            node.pos_start, node.pos_end,
            "Expected list or dictionary for indexing",
            context
        ))

#######################################
# RUN
#######################################

global_symbol_table = SymbolTable()

global_symbol_table.set("null", Number.null)
global_symbol_table.set("false", Number.false)
global_symbol_table.set("true", Number.true)


global_symbol_table.set("serveChai", BuiltInFunction.print)
global_symbol_table.set("serve_ret", BuiltInFunction.print_ret)
global_symbol_table.set("orderChai", BuiltInFunction.input)
global_symbol_table.set("orderChai_int", BuiltInFunction.input_int)
global_symbol_table.set("clean", BuiltInFunction.clear)
global_symbol_table.set("cls", BuiltInFunction.clear)
global_symbol_table.set("type_chai", BuiltInFunction("type"))
global_symbol_table.set("current_date_time", BuiltInFunction("current_date_time"))
global_symbol_table.set("list_len", BuiltInFunction.list_len)
global_symbol_table.set("str_len", BuiltInFunction.str_len)
global_symbol_table.set("len", BuiltInFunction.len)
global_symbol_table.set("int", BuiltInFunction("int"))
global_symbol_table.set("str", BuiltInFunction("str"))
global_symbol_table.set("float", BuiltInFunction("float"))
global_symbol_table.set("time_freeze", BuiltInFunction("time_freeze"))

global_symbol_table.set("is_chai_num", BuiltInFunction.is_number)
global_symbol_table.set("is_chai_str", BuiltInFunction.is_string)
global_symbol_table.set("is_chai_list", BuiltInFunction.is_list)
global_symbol_table.set("is_brew", BuiltInFunction.is_function)
global_symbol_table.set("is_empty", BuiltInFunction("is_empty"))


global_symbol_table.set("append", BuiltInFunction.append)
global_symbol_table.set("pop", BuiltInFunction.pop)
global_symbol_table.set("extend", BuiltInFunction.extend)
global_symbol_table.set("sort_list", BuiltInFunction("sort"))
global_symbol_table.set("reverse_list", BuiltInFunction("reverse"))
global_symbol_table.set("range", BuiltInFunction("range"))
global_symbol_table.set("element_at", BuiltInFunction("element_at"))
global_symbol_table.set("join_to_str", BuiltInFunction("join"))


global_symbol_table.set("run", BuiltInFunction.run)

global_symbol_table.set("to_lower", BuiltInFunction("to_lower"))
global_symbol_table.set("to_upper", BuiltInFunction("to_upper"))
global_symbol_table.set("char_at", BuiltInFunction("char_at"))
global_symbol_table.set("index_of", BuiltInFunction("index_of"))
global_symbol_table.set("slice", BuiltInFunction("slice"))
global_symbol_table.set("starts_with", BuiltInFunction("starts_with"))
global_symbol_table.set("ends_with", BuiltInFunction("ends_with"))
global_symbol_table.set("trim", BuiltInFunction("trim"))
global_symbol_table.set("replace", BuiltInFunction("replace"))
global_symbol_table.set("split_to_list", BuiltInFunction("split"))
global_symbol_table.set("capitalize", BuiltInFunction("capitalize"))
global_symbol_table.set("title", BuiltInFunction("title"))
global_symbol_table.set("is_equals", BuiltInFunction("is_equals"))


global_symbol_table.set("Math_PI", Number.math_PI)
global_symbol_table.set("Math_E", Number.math_E)

global_symbol_table.set("Math_sin", BuiltInFunction("sin"))
global_symbol_table.set("Math_cos", BuiltInFunction("cos"))
global_symbol_table.set("Math_tan", BuiltInFunction("tan"))
global_symbol_table.set("Math_cot", BuiltInFunction("cot"))
global_symbol_table.set("Math_sec", BuiltInFunction("sec"))
global_symbol_table.set("Math_cosec", BuiltInFunction("cosec"))

global_symbol_table.set("Math_arcsin", BuiltInFunction("arcsin"))
global_symbol_table.set("Math_arccos", BuiltInFunction("arccos"))
global_symbol_table.set("Math_arctan", BuiltInFunction("arctan"))
global_symbol_table.set("Math_arccot", BuiltInFunction("arccot"))
global_symbol_table.set("Math_arccosec", BuiltInFunction("arccosec"))
global_symbol_table.set("Math_arcsec", BuiltInFunction("arcsec"))

global_symbol_table.set("Math_sinh", BuiltInFunction("sinh"))
global_symbol_table.set("Math_cosh", BuiltInFunction("cosh"))
global_symbol_table.set("Math_tanh", BuiltInFunction("tanh"))
global_symbol_table.set("Math_sech", BuiltInFunction("sech"))
global_symbol_table.set("Math_cosech", BuiltInFunction("cosech"))
global_symbol_table.set("Math_coth", BuiltInFunction("coth"))

global_symbol_table.set("Math_arcsinh", BuiltInFunction("arcsinh"))
global_symbol_table.set("Math_arccosh", BuiltInFunction("arccosh"))
global_symbol_table.set("Math_arctanh", BuiltInFunction("arctanh"))
global_symbol_table.set("Math_arcsech", BuiltInFunction("arcsech"))
global_symbol_table.set("Math_arccosech", BuiltInFunction("arccosech"))
global_symbol_table.set("Math_arccoth", BuiltInFunction("arccot"))

global_symbol_table.set("Math_factorial", BuiltInFunction("factorial"))
global_symbol_table.set("Math_sqrt", BuiltInFunction("sqrt"))
global_symbol_table.set("Math_floor", BuiltInFunction("floor"))
global_symbol_table.set("Math_ceil", BuiltInFunction("ceil"))
global_symbol_table.set("Math_round", BuiltInFunction("round"))
global_symbol_table.set("Math_absoluteval", BuiltInFunction("abs"))
global_symbol_table.set("Math_random_chai", BuiltInFunction("randint"))
global_symbol_table.set("Math_mod_div", BuiltInFunction("mod_div"))

global_symbol_table.set("dict_has", BuiltInFunction("has"))
global_symbol_table.set("dict_keys", BuiltInFunction("keys"))
global_symbol_table.set("dict_values", BuiltInFunction("values"))
global_symbol_table.set("dict_items", BuiltInFunction("items"))
global_symbol_table.set("dict_get", BuiltInFunction("get"))
global_symbol_table.set("dict_pop", BuiltInFunction("pop"))
global_symbol_table.set("dict_update", BuiltInFunction("update"))
global_symbol_table.set("dict_clear", BuiltInFunction("clear"))
global_symbol_table.set("dict_copy", BuiltInFunction("copy"))






def run(fn, text):
  
  
  # Generate tokens
  lexer = Lexer(fn, text)
  tokens, error = lexer.make_tokens()
  if error: return None, error
  
  # Generate AST
  parser = Parser(tokens)
  ast = parser.parse()
  if ast.error: return None, ast.error

  # Run program
  interpreter = Interpreter()
  context = Context('<program>')
  context.symbol_table = global_symbol_table
  result = interpreter.visit(ast.node, context)

  return result.value, result.error



def run_web(text: str):
    """
    Web-safe execution wrapper around run()
    Captures printed output AND execution trace
    """

    import sys
    import io

    # reset execution trace
    EXECUTION_TRACE.clear()

    # reset global symbol table
    global global_symbol_table
    global_symbol_table = SymbolTable()

    # =========================
    # CORE CONSTANTS
    # =========================
    global_symbol_table.set("null", Number.null)
    global_symbol_table.set("false", Number.false)
    global_symbol_table.set("true", Number.true)

    # =========================
    # BASIC I/O
    # =========================
    global_symbol_table.set("serveChai", BuiltInFunction.print)
    global_symbol_table.set("serve_ret", BuiltInFunction.print_ret)
    global_symbol_table.set("orderChai", BuiltInFunction.input)
    global_symbol_table.set("orderChai_int", BuiltInFunction.input_int)

    # =========================
    # SYSTEM
    # =========================
    global_symbol_table.set("clean", BuiltInFunction.clear)
    global_symbol_table.set("cls", BuiltInFunction.clear)
    global_symbol_table.set("type_chai", BuiltInFunction("type"))
    global_symbol_table.set("current_date_time", BuiltInFunction("current_date_time"))
    global_symbol_table.set("time_freeze", BuiltInFunction("time_freeze"))

    # =========================
    # LENGTH / TYPE
    # =========================
    global_symbol_table.set("list_len", BuiltInFunction.list_len)
    global_symbol_table.set("str_len", BuiltInFunction.str_len)
    global_symbol_table.set("len", BuiltInFunction.len)

    global_symbol_table.set("int", BuiltInFunction("int"))
    global_symbol_table.set("str", BuiltInFunction("str"))
    global_symbol_table.set("float", BuiltInFunction("float"))

    # =========================
    # TYPE CHECK
    # =========================
    global_symbol_table.set("is_chai_num", BuiltInFunction.is_number)
    global_symbol_table.set("is_chai_str", BuiltInFunction.is_string)
    global_symbol_table.set("is_chai_list", BuiltInFunction.is_list)
    global_symbol_table.set("is_brew", BuiltInFunction.is_function)
    global_symbol_table.set("is_empty", BuiltInFunction("is_empty"))

    # =========================
    # LIST FUNCTIONS
    # =========================
    global_symbol_table.set("append", BuiltInFunction.append)
    global_symbol_table.set("pop", BuiltInFunction.pop)
    global_symbol_table.set("extend", BuiltInFunction.extend)
    global_symbol_table.set("sort_list", BuiltInFunction("sort"))
    global_symbol_table.set("reverse_list", BuiltInFunction("reverse"))
    global_symbol_table.set("range", BuiltInFunction("range"))
    global_symbol_table.set("element_at", BuiltInFunction("element_at"))
    global_symbol_table.set("join_to_str", BuiltInFunction("join"))

    # =========================
    # RUN FILE
    # =========================
    global_symbol_table.set("run", BuiltInFunction.run)

    # =========================
    # STRING FUNCTIONS
    # =========================
    global_symbol_table.set("to_lower", BuiltInFunction("to_lower"))
    global_symbol_table.set("to_upper", BuiltInFunction("to_upper"))
    global_symbol_table.set("char_at", BuiltInFunction("char_at"))
    global_symbol_table.set("index_of", BuiltInFunction("index_of"))
    global_symbol_table.set("slice", BuiltInFunction("slice"))
    global_symbol_table.set("starts_with", BuiltInFunction("starts_with"))
    global_symbol_table.set("ends_with", BuiltInFunction("ends_with"))
    global_symbol_table.set("trim", BuiltInFunction("trim"))
    global_symbol_table.set("replace", BuiltInFunction("replace"))
    global_symbol_table.set("split_to_list", BuiltInFunction("split"))
    global_symbol_table.set("capitalize", BuiltInFunction("capitalize"))
    global_symbol_table.set("title", BuiltInFunction("title"))
    global_symbol_table.set("is_equals", BuiltInFunction("is_equals"))

    # =========================
    # MATH CONSTANTS
    # =========================
    global_symbol_table.set("Math_PI", Number.math_PI)
    global_symbol_table.set("Math_E", Number.math_E)

    # =========================
    # TRIGONOMETRY
    # =========================
    global_symbol_table.set("Math_sin", BuiltInFunction("sin"))
    global_symbol_table.set("Math_cos", BuiltInFunction("cos"))
    global_symbol_table.set("Math_tan", BuiltInFunction("tan"))
    global_symbol_table.set("Math_cot", BuiltInFunction("cot"))
    global_symbol_table.set("Math_sec", BuiltInFunction("sec"))
    global_symbol_table.set("Math_cosec", BuiltInFunction("cosec"))

    global_symbol_table.set("Math_arcsin", BuiltInFunction("arcsin"))
    global_symbol_table.set("Math_arccos", BuiltInFunction("arccos"))
    global_symbol_table.set("Math_arctan", BuiltInFunction("arctan"))
    global_symbol_table.set("Math_arccot", BuiltInFunction("arccot"))
    global_symbol_table.set("Math_arccosec", BuiltInFunction("arccosec"))
    global_symbol_table.set("Math_arcsec", BuiltInFunction("arcsec"))

    # =========================
    # HYPERBOLIC
    # =========================
    global_symbol_table.set("Math_sinh", BuiltInFunction("sinh"))
    global_symbol_table.set("Math_cosh", BuiltInFunction("cosh"))
    global_symbol_table.set("Math_tanh", BuiltInFunction("tanh"))
    global_symbol_table.set("Math_sech", BuiltInFunction("sech"))
    global_symbol_table.set("Math_cosech", BuiltInFunction("cosech"))
    global_symbol_table.set("Math_coth", BuiltInFunction("coth"))

    global_symbol_table.set("Math_arcsinh", BuiltInFunction("arcsinh"))
    global_symbol_table.set("Math_arccosh", BuiltInFunction("arccosh"))
    global_symbol_table.set("Math_arctanh", BuiltInFunction("arctanh"))
    global_symbol_table.set("Math_arcsech", BuiltInFunction("arcsech"))
    global_symbol_table.set("Math_arccosech", BuiltInFunction("arccosech"))
    global_symbol_table.set("Math_arccoth", BuiltInFunction("arccot"))

    # =========================
    # MATH UTILITIES
    # =========================
    global_symbol_table.set("Math_factorial", BuiltInFunction("factorial"))
    global_symbol_table.set("Math_sqrt", BuiltInFunction("sqrt"))
    global_symbol_table.set("Math_floor", BuiltInFunction("floor"))
    global_symbol_table.set("Math_ceil", BuiltInFunction("ceil"))
    global_symbol_table.set("Math_round", BuiltInFunction("round"))
    global_symbol_table.set("Math_absoluteval", BuiltInFunction("abs"))
    global_symbol_table.set("Math_random_chai", BuiltInFunction("randint"))
    global_symbol_table.set("Math_mod_div", BuiltInFunction("mod_div"))

    # =========================
    # DICTIONARY
    # =========================
    global_symbol_table.set("dict_has", BuiltInFunction("has"))
    global_symbol_table.set("dict_keys", BuiltInFunction("keys"))
    global_symbol_table.set("dict_values", BuiltInFunction("values"))
    global_symbol_table.set("dict_items", BuiltInFunction("items"))
    global_symbol_table.set("dict_get", BuiltInFunction("get"))
    global_symbol_table.set("dict_pop", BuiltInFunction("pop"))
    global_symbol_table.set("dict_update", BuiltInFunction("update"))
    global_symbol_table.set("dict_clear", BuiltInFunction("clear"))
    global_symbol_table.set("dict_copy", BuiltInFunction("copy"))

    # =========================
    # CAPTURE OUTPUT
    # =========================
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()

    try:
        value, error = run("<web>", text)

        if error:
            return error.as_string(), EXECUTION_TRACE

        printed_output = buffer.getvalue()

        if printed_output.strip():
            return printed_output, EXECUTION_TRACE

        if value is not None:
            return str(value), EXECUTION_TRACE

        return "null", EXECUTION_TRACE

    except Exception as e:
        return f"[Internal Error] {e}", EXECUTION_TRACE

    finally:
        sys.stdout = old_stdout