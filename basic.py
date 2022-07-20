####################################
# IMPORT
####################################

from strings_with_arrows import *
import string

####################################
# CHARS
####################################

DIGITS = '0123456789'
LETTERS = string.ascii_letters
LETTERS_DIGITS = LETTERS + DIGITS


####################################
# ERROR CLASS
####################################

class Error:
    def __init__(self, pos_start_, pos_end_, error_name_, details_):
        self.pos_start = pos_start_
        self.pos_end = pos_end_
        self.error_name = error_name_
        self.details = details_

    def as_string(self):
        result = f"{self.error_name}: {self.details}"
        result += f'File {self.pos_start.fn}, line {self.pos_start.ln + 1}'
        result += '\n\n' + string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)
        return result


class UnknownCharacterError(Error):
    def __init__(self, pos_start_, pos_end_, details_):
        super().__init__(pos_start_, pos_end_, 'UnknownCharacterError: character not defined', details_)


class VariableNotDefinedError(Error):
    def __init__(self, pos_start_, pos_end_, details_):
        super().__init__(pos_start_, pos_end_, 'VariableNotDefinedError: variable is not previously defined', details_)


class InvalidSyntaxError(Error):
    def __init__(self, pos_start_, pos_end_, details_):
        super().__init__(pos_start_, pos_end_, 'InvalidSyntaxError: failed to interpret parser', details_)


class RuntimeError(Error):
    def __init__(self, pos_start_, pos_end_, details_, context_):
        super().__init__(pos_start_, pos_end_, 'RuntimeError', details_)
        self.context = context_

    def as_string(self):
        result = self.generate_traceback()
        result += f'{self.error_name}: {self.details}'
        result += '\n\n' + string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)
        return result

    def generate_traceback(self):
        result = ''
        pos = self.pos_start
        ctxt = self.context

        while ctxt:
            result = f' File {pos.fn}, line {pos.ln + 1}, in {ctxt.display_name}\n' + result
            pos = ctxt.parent_entry_pos
            ctxt = ctxt.parent

        return 'Traced to: \n' + result


class RuntimeResult:
    def __init__(self):
        self.value = None
        self.error = None

    def register(self, res):
        if res.error:
            self.error = res.error
        return res.value

    def success(self, value):
        self.value = value
        return self

    def failure(self, error):
        self.error = error
        return self


####################################
# POSITION
####################################

class Position:
    def __init__(self, idx_, ln_, col_, fn_, ftxt_):
        self.idx = idx_
        self.ln = ln_
        self.col = col_
        self.fn = fn_
        self.ftxt = ftxt_

    def advance(self, current_char=None):
        self.idx += 1
        self.col += 1

        if current_char == '\n':
            self.ln += 1
            self.col = 0

        return self

    def copy(self):
        return Position(self.idx, self.ln, self.col, self.fn, self.ftxt)


####################################
# TOKENS
####################################

TT_INT = 'INT'
TT_FLOAT = 'FLOAT'
TT_PLUS = 'PLUS'
TT_MINUS = 'MINUS'
TT_MULT = 'MULT'
TT_DIV = 'DIV'
TT_POW = 'POW'
TT_LPAREN = 'LPAREN'
TT_RPAREN = 'RPAREN'
TT_EOF = 'EOF'
TT_IDENTIFIER = 'ID'
TT_KEYWORD = 'KEYWORDS'
TT_EQ = '='

KEYWORDS = [
    'head'
]


class Token:
    def __init__(self, type_, value_=None, pos_start_=None, pos_end_=None):
        self.type = type_
        self.value = value_

        if pos_start_:
            self.pos_start = pos_start_.copy()
            self.pos_end = pos_start_.copy()
            self.pos_end.advance()

        if pos_end_:
            self.pos_end = pos_end_.copy()

    def matches(self, type_, value_):
        return self.type == type_ and self.value == value_

    def __repr__(self):
        if self.value:
            return f'{self.type}:{self.value}'
        return f'{self.type}'


####################################
# LEXER
####################################

class Lexer:
    def __init__(self, text_, fn_):
        self.fn = fn_
        self.text = text_
        self.pos = Position(-1, 0, -1, fn_, text_)
        self.current_char = None
        self.advance()

    def advance(self):
        self.pos.advance(self.current_char)
        self.current_char = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None

    def make_token(self):
        tokens = []
        while self.current_char != None:
            if self.current_char in ' \t':
                self.advance()
            elif self.current_char in DIGITS:
                tokens.append(self.make_number())
            elif self.current_char == '+':
                tokens.append(Token(TT_PLUS, pos_start_=self.pos))
                self.advance()
            elif self.current_char == '-':
                tokens.append(Token(TT_MINUS, pos_start_=self.pos))
                self.advance()
            elif self.current_char == '*':
                tokens.append(Token(TT_MULT, pos_start_=self.pos))
                self.advance()
            elif self.current_char == '/':
                tokens.append(Token(TT_DIV, pos_start_=self.pos))
                self.advance()
            elif self.current_char == '^':
                tokens.append(Token(TT_POW, pos_start_=self.pos))
                self.advance()
            elif self.current_char == '(':
                tokens.append(Token(TT_LPAREN, pos_start_=self.pos))
                self.advance()
            elif self.current_char == ')':
                tokens.append(Token(TT_RPAREN, pos_start_=self.pos))
                self.advance()
            elif self.current_char == '=':
                tokens.append(Token(TT_EQ, pos_start_=self.pos))
                self.advance()
            elif self.current_char in LETTERS:
                tokens.append(self.make_identifier())
                self.advance()
            else:
                pos_start = self.pos.copy()
                char = self.current_char
                self.advance()
                return [], UnknownCharacterError(pos_start, self.pos, "'" + char + "' \n")

        tokens.append(Token(TT_EOF, pos_start_=self.pos))
        return tokens, None

    def make_number(self):
        num_str = ''
        dot_count = 0
        pos_start = self.pos.copy()

        while self.current_char != None and self.current_char in DIGITS + '.':
            if self.current_char == '.':
                if dot_count == 1: break
                dot_count += 1
                num_str += '.'
            else:
                num_str += self.current_char
            self.advance()

        if dot_count == 0:
            return Token(TT_INT, int(num_str), pos_start_=pos_start, pos_end_=self.pos)
        else:
            return Token(TT_FLOAT, float(num_str), pos_start_=pos_start, pos_end_=self.pos)

    def make_identifier(self):
        id_str = ''
        pos_start = self.pos.copy()

        while self.current_char is not None and self.current_char in LETTERS_DIGITS + '_':
            id_str += self.current_char
            self.advance()

        token_type = TT_KEYWORD if id_str in KEYWORDS else TT_IDENTIFIER
        return Token(token_type, id_str, pos_start, self.pos)


####################################
# NODES
####################################

class NumberNode:
    def __init__(self, token_):
        self.token = token_

        self.pos_start = self.token.pos_start
        self.pos_end = self.token.pos_end

    def __repr__(self):
        return f'{self.token}'


class BinOpNode:
    def __init__(self, left_node_, op_token_, right_node_):
        self.left_node = left_node_
        self.op_token = op_token_
        self.right_node = right_node_

        self.pos_start = self.left_node.pos_start
        self.pos_end = self.right_node.pos_end

    def __repr__(self):
        return f'({self.left_node}, {self.op_token}, {self.right_node})'


class UnaryOpNode:
    def __init__(self, op_token_, node_):
        self.op_token = op_token_
        self.node = node_

        self.pos_start = self.op_token.pos_start
        self.pos_end = node_.pos_end

    def __repr__(self):
        return f'{self.op_token}, {self.node}'


class VarAccessNode:
    def __init__(self, var_name_token_):
        self.var_name_token = var_name_token_
        self.pos_start = self.var_name_token.pos_start
        self.pos_end = self.var_name_token.pos_end


class VarAssignNode:
    def __init__(self, var_name_token_, value_node_):
        self.var_name_token = var_name_token_
        self.value_node = value_node_

        self.pos_start = self.var_name_token.pos_start
        self.pos_end = self.value_node.pos_end


####################################
# PARSER
####################################

class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None
        self.advance_count = 0

    def register2(self):
        self.advance_count += 1

    def register(self, res_):
        self.advance_count += res_.advance_count
        if res_.error:
            self.error = res_.error
        return res_.node

        return res_

    def success(self, node_):
        self.node = node_
        return self

    def failure(self, error_):
        if not self.error or self.advance_count == 0:
            self.error = error_
        return self


class Parser:
    def __init__(self, tokens_):
        self.tokens = tokens_
        self.token_idx = -1
        self.advance()

    def advance(self):
        self.token_idx += 1
        if self.token_idx < len(self.tokens):
            self.current_token = self.tokens[self.token_idx]

        return self.current_token

    ###########################

    def parse(self):
        res = self.expr()
        if not res.error and self.current_token.type != TT_EOF:
            return res.failure(InvalidSyntaxError(
                self.current_token.pos_start, self.current_token.pos_end, "Expected '+', '-', '*' or '/' \n"
            ))
        return res

    def atom(self):
        res = ParseResult()
        token = self.current_token

        if token.type in (TT_INT, TT_FLOAT):
            res.register2()
            self.advance()
            return res.success(NumberNode(token))

        elif token.type == TT_IDENTIFIER:
            res.register2()
            self.advance()
            return res.success(VarAccessNode(token))

        elif token.type == TT_LPAREN:
            res.register2()
            self.advance()
            expr = res.register(self.expr())
            if res.error:
                return res
            if self.current_token.type == TT_RPAREN:
                res.register2()
                self.advance()
                return res.success(expr)
            else:
                return res.failure(InvalidSyntaxError(self.current_token.pos_start,
                                                      self.current_token.pos_end,
                                                      "Expected ')' \n"))
        return res.failure(InvalidSyntaxError(
            token.pos_start, token.pos_end, "Expected int, float, variable identifier, '+', '-' or '(' \n"
        ))

    def power(self):
        return self.bin_op(self.atom, (TT_POW,))

    def factor(self):
        res = ParseResult()
        token = self.current_token

        if token.type in (TT_PLUS, TT_MINUS):
            res.register2()
            self.advance()
            factor = res.register(self.factor())
            if res.error:
                return res
            return res.success(UnaryOpNode(token, factor))

        return self.power()

    def bin_op(self, fn_, ops_):
        res = ParseResult()
        left = res.register(fn_())
        if res.error:
            return res

        while self.current_token.type in ops_:
            op_token = self.current_token
            res.register2()
            self.advance()
            right = res.register(fn_())
            if res.error:
                return res
            left = BinOpNode(left, op_token, right)

        return res.success(left)

    def expr(self):
        res = ParseResult()
        if self.current_token.matches(TT_KEYWORD, 'head'):
            res.register2()
            self.advance()

            if self.current_token.type != TT_IDENTIFIER:
                return res.failure(InvalidSyntaxError(
                    self.current_token.pos_start, self.current_token.pos_end,
                    "Expected a variable identifier."
                ))
            var_name = self.current_token
            res.register2()
            self.advance()

            if self.current_token.type != TT_EQ:
                return res.failure(InvalidSyntaxError(self.current_token.pos_start, self.current_token.pos_end,
                                                      "Expected '=' sign."))

            res.register2()
            self.advance()
            expr = res.register(self.expr())
            if res.error:
                return res
            return res.success(VarAssignNode(var_name, expr))
        node = res.register(self.bin_op(self.term, (TT_PLUS, TT_MINUS)))

        if res.error:
            return res.failure(InvalidSyntaxError(self.current_token.pos_start, self.current_token.pos_end,
                                                  "Expected 'head', int, float, variable identifier, '+', '-',"
                                                  " or '('. "))

        return res.success(node)

    def term(self):
        return self.bin_op(self.factor, (TT_MULT, TT_DIV))


####################################
# CONTEXT
####################################

class Context:
    def __init__(self, display_name_, parent=None, parent_entry_pos=None):
        self.display_name = display_name_
        self.parent = parent
        self.parent_entry_pos = parent_entry_pos
        self.symbol_table = None


####################################
# SYMBOLS
####################################

class SymbolTable:
    def __init__(self):
        self.symbols = {}
        self.parent = None

    def get(self, name_):
        value = self.symbols.get(name_, None)
        if value == None and self.parent:
            return self.parent.get(name_)
        return value

    def set(self, name_, value_):
        self.symbols[name_] = value_

    def remove(self, name_):
        del self.symbols[name_]


####################################
# INTERPRETER
####################################

class Interpreter:
    def visit(self, node_, context_):
        method_name = f'visit_{type(node_).__name__}'
        method = getattr(self, method_name, self.no_visit_method)
        # print(type(node_))
        return method(node_, context_)

    def no_visit_method(self, node_, context_):
        raise Exception(f'No visit_{type(node_).__name__} method defined.')

    def visit_NumberNode(self, node_, context_):
        number = Number(node_.token.value)
        number.set_context(context_).set_pos(node_.pos_start, node_.pos_end)
        return RuntimeResult().success(number)

    def visit_BinOpNode(self, node_, context_):
        res = RuntimeResult()
        left = res.register(self.visit(node_.left_node, context_))
        if res.error:
            return res
        right = res.register(self.visit(node_.right_node, context_))
        if res.error:
            return res

        right.set_pos(node_.right_node.pos_start, node_.right_node.pos_end)
        left.set_pos(node_.left_node.pos_start, node_.left_node.pos_end)

        if node_.op_token.type == TT_PLUS:
            result, error = left.added_to(right)
        elif node_.op_token.type == TT_MINUS:
            result, error = left.subtracted_by(right)
        elif node_.op_token.type == TT_MULT:
            result, error = left.multiplied_by(right)
        elif node_.op_token.type == TT_DIV:
            result, error = left.divided_by(right)
        elif node_.op_token.type == TT_POW:
            result, error = left.to_the_power_of(right)

        if error:
            return res.failure(error)
        else:
            result.set_pos(node_.pos_start, node_.pos_end)
            return res.success(result)

    def visit_VarAccessNode(self, node_, context_):
        res = RuntimeResult()
        var_name = node_.var_name_token.value
        value = context_.symbol_table.get(var_name)


        if not value:
            return res.failure(RuntimeError(
                node_.pos_start, node_.pos_end,
                f"'{var_name}' was never declared.",
                context_
            ))

        value = value.copy().set_pos(node_.pos_start, node_.pos_end)
        return res.success(value)

    def visit_VarAssignNode(self, node_, context_):
        res = RuntimeResult()
        var_name = node_.var_name_token.value
        value = res.register(self.visit(node_.value_node, context_))
        if res.error:
            return res

        context_.symbol_table.set(var_name, value)
        return res.success(value)

    def visit_UnaryOpNode(self, node_, context_):
        res = RuntimeResult()
        number = res.register(self.visit(node_.node, context_))
        if res.error:
            return res

        error = None

        if node_.op_token.type == TT_MINUS:
            number, error = number.multiplied_by(Number(-1))

        number.set_pos(node_.pos_start, node_.pos_end)

        if error:
            return res.failure(error)
        else:
            return res.success(number)


####################################
# VALUES
####################################

class Number:
    def __init__(self, value_):
        self.value = value_
        self.set_pos()
        self.set_context()

    def set_pos(self, pos_start=None, pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self

    def set_context(self, context_=None):
        self.context = context_
        return self

    def added_to(self, other_):
        if isinstance(other_, Number):
            op_result = Number(self.value + other_.value)
            op_result.set_context(self.context)
            return op_result, None

    def subtracted_by(self, other_):
        if isinstance(other_, Number):
            op_result = Number(self.value - other_.value)
            op_result.set_context(self.context)
            return op_result, None

    def multiplied_by(self, other_):
        if isinstance(other_, Number):
            op_result = Number(self.value * other_.value)
            op_result.set_context(self.context)
            return op_result, None

    def to_the_power_of(self, other_):
        if isinstance(other_, Number):
            op_result = Number(self.value ** other_.value)
            op_result.set_context(self.context)
            return op_result, None

    def divided_by(self, other_):
        if isinstance(other_, Number):
            if other_.value == 0:
                return [], RuntimeError(other_.pos_start, other_.pos_end, "Division by zero. \n", self.context)
            op_result = Number(self.value / other_.value)
            op_result.set_context(self.context)
            return op_result, None

    def copy(self):
        copy = Number(self.value)
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.context)
        return copy

    def __repr__(self):
        return str(self.value)


####################################
# RUN
####################################

global_symbol_table = SymbolTable()
global_symbol_table.set("null", Number(0))


def run(text, fn):
    lexer = Lexer(fn, text)
    tokens, error = lexer.make_token()

    if error:
        return None, error

    parser = Parser(tokens)
    ast = parser.parse()

    if ast.error:
        return None, ast.error

    interpreter = Interpreter()
    context = Context('<program>')
    context.symbol_table = global_symbol_table
    result = interpreter.visit(ast.node, context)

    return result.value, result.error
