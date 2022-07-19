####################################
# IMPORT
####################################

from strings_with_arrows import *

####################################
# CHARS
####################################

DIGITS ='0123456789'
LETTERS = 'abcdefghijklmnopqrstuvxwyz'

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
        result = f'{self.error_name}: {self.details}'
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
    def __init__(self, pos_start_, pos_end_, details_):
        super().__init__(pos_start_, pos_end_, 'RuntimeError', details_)

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
TT_LPAREN = 'LPAREN'
TT_RPAREN = 'RPAREN'
TT_EOF = 'EOF'

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
            elif self.current_char == '(':
                tokens.append(Token(TT_LPAREN, pos_start_=self.pos))
                self.advance()
            elif self.current_char == ')':
                tokens.append(Token(TT_RPAREN, pos_start_=self.pos))
                self.advance()
            elif self.current_char in LETTERS:
                pos_start = self.pos.copy()
                char = self.current_char
                self.advance()
                return [], VariableNotDefinedError(pos_start, self.pos, "'" + char + "' \n")
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


####################################
# PARSER
####################################

class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None

    def register(self, res_):
        if isinstance(res_, ParseResult):
            if res_.error:
                self.error = res_.error
            return res_.node

        return res_

    def success(self, node_):
        self.node = node_
        return self

    def failure(self, error_):
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

    def factor(self):
        res = ParseResult()
        token = self.current_token

        if token.type in (TT_PLUS, TT_MINUS):
            res.register(self.advance())
            factor = res.register(self.factor())
            if res.error:
                return res
            return res.success(UnaryOpNode(token, factor))

        elif token.type in (TT_INT, TT_FLOAT):
            res.register(self.advance())
            return res.success(NumberNode(token))

        elif token.type == TT_LPAREN:
            res.register(self.advance())
            expr = res.register(self.expr())
            if res.error:
                return res
            if self.current_token.type == TT_RPAREN:
                res.register(self.advance())
                return res.success(expr)
            else:
                return res.failure(InvalidSyntaxError(self.current_token.pos_start,
                                   self.current_token.pos_end,
                                   "Expected ')' \n"))

        return res.failure(InvalidSyntaxError(
            token.pos_start, token.pos_end, "Expected int, float \n"
        ))


    def bin_op(self, fn_, ops_):
        res = ParseResult()
        left = res.register(fn_())
        if res.error:
            return res


        while self.current_token.type in ops_:
            op_token = self.current_token
            res.register(self.advance())
            right = res.register(fn_())
            if res.error:
                return res
            left = BinOpNode(left, op_token, right)

        return res.success(left)

    def expr(self):
        return self.bin_op(self.term, (TT_PLUS, TT_MINUS))

    def term(self):
        return self.bin_op(self.factor, (TT_MULT, TT_DIV))

####################################
# INTERPRETER
####################################

class Interpreter:
    def visit(self, node_):
        method_name = f'visit_{type(node_).__name__}'
        method = getattr(self, method_name, self.no_visit_method)
        #print(type(node_))
        return method(node_)

    def no_visit_method(self, node_):
        raise Exception(f'No visit_{type(node_).__name__} method defined.')

    def visit_NumberNode(self, node_):
        return RuntimeResult().success(Number(node_.token.value))

    def visit_BinOpNode(self, node_):
        res = RuntimeResult()
        left = res.register(self.visit(node_.left_node))
        if res.error:
            return res
        right = res.register(self.visit(node_.right_node))
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




        if error:
            return res.failure(error)
        else:
            result.set_pos(node_.pos_start, node_.pos_end)
            return res.success(result)


    def visit_UnaryOpNode(self, node_):
        res = RuntimeResult()
        number = res.register(self.visit(node_.node))
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

    def set_pos(self, pos_start=None, pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end

    def added_to(self, other_):
        if isinstance(other_, Number):
            return Number(self.value + other_.value), None

    def subtracted_by(self, other_):
        if isinstance(other_, Number):
            return Number(self.value - other_.value), None

    def multiplied_by(self, other_):
        if isinstance(other_, Number):
            return Number(self.value * other_.value), None

    def divided_by(self, other_):
        if isinstance(other_, Number):
            if other_.value == 0:
                #print(RuntimeError(other_.pos_start, other_.pos_end, "Division by zero."))
                return [], RuntimeError(other_.pos_start, other_.pos_end, "Division by zero. \n")
            return Number(self.value / other_.value), None

    def __repr__(self):
        return str(self.value)

####################################
# RUN
####################################

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
    result = interpreter.visit(ast.node)


    return result.value, result.error