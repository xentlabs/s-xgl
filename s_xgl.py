"""
	The interpreter will accept anything; the only condition to be a valid programs is that we end with an empty line.
"""
from dataclasses import dataclass, field
import argparse
import json
import math
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_LLM_SERVER_URL = "http://127.0.0.1:30000"
DEFAULT_GAME_PATH = os.path.join(os.path.dirname(__file__), "games", "Condense.sxgl")

@dataclass
class M:
	""" Model runtime state  """
	url: str = DEFAULT_LLM_SERVER_URL
	context: list[int] = field(default_factory=list)
	score_acc: float = 0.0
	xent_acc: float = 0.0


@dataclass
class S:
	""" Token string runtime state """
	tokens: list[int] = field(default_factory=list)

	def __len__(self): return len(self.tokens)
	def __repr__(self): return f"{self.tokens}"


class Env:
	def __init__(self,
			print_strings=True,
			number_token_strings=8,
			max_token_string_length=256,
			ensure_factor=16,
			m_initial="m",
			s_initial="s"):
		self.print_strings = print_strings
		self.max_len = max_token_string_length
		self.ens_fac = ensure_factor
		(self.m_initial, self.s_initial, self.v_initial) = (m_initial, s_initial, m_initial + s_initial)
		default_player_model = M()
		default_judge_model = M()
		self.default_data_model = M()
		self.m_dict = dict(mp=default_player_model, mj=default_judge_model, md=self.default_data_model)
		self.make_s_dict(number_token_strings)
		print(self.format_s_dict())
		self.v_dict = self.m_dict | self.s_dict
		self.prev_line = ""
		self.prev_line_tokens = []

	def make_s_dict(self, number_token_strings):
		num_s_digits = max(1, math.ceil(math.log10(number_token_strings)))
		# If e.g. num_s_digits=3, make the variable names s000, s001, s002
		self.s_dict = {f"{self.s_initial}{i:0{num_s_digits}}": S() for i in range(number_token_strings)}

	def ensure_nonlinearity(self, xent_acc):
		return xent_acc / self.ens_fac if xent_acc > 0 else xent_acc * self.ens_fac

	def post_json(self, url, path, payload):
		request = Request(
			url=f"{url.rstrip('/')}{path}",
			data=json.dumps(payload).encode("utf-8"),
			headers={"Content-Type": "application/json"},
			method="POST",
		)
		try:
			with urlopen(request, timeout=60) as response:
				return json.loads(response.read().decode("utf-8"))
		except HTTPError as exc:
			body = exc.read().decode("utf-8", errors="replace")
			raise RuntimeError(f"LLM server error for {path}: HTTP {exc.code}: {body}") from exc
		except URLError as exc:
			raise RuntimeError(f"Could not reach LLM server at {url}: {exc.reason}") from exc

	def tokenize(self, char_string):
		response = self.post_json(self.default_data_model.url, "/tokenize", {"text": char_string})
		return response["tokens"]

	def detokenize(self, tokens):
		response = self.post_json(self.default_data_model.url, "/detokenize", {"tokens": tokens})
		return response["text"]

	def format_tokens(self, tokens):
		return repr(self.detokenize(tokens)) if self.print_strings else f"{tokens}"

	def format_s_dict(self):
		if not self.print_strings: return self.s_dict
		return {name: self.detokenize(s.tokens) for name, s in self.s_dict.items()}

	def clear_model(self, m):
		m.context = []
		m.score_acc = 0.0
		m.xent_acc = 0.0

	def clear_string(self, s): s.tokens = []

	def run(self, program): self.run_lines(program.split("\n"))

	def run_lines(self, lines):
		if len(lines) == 0 or len(lines[-1]) != 0: lines = lines + [""] # add an empty line if the last line is not empty
		for line in lines:
			self.run_line(line)
			self.prev_line = line

	def run_line(self, line):
		(ip, left_operand, right_operand) = self.extract_instruction(line)
		if len(line) == 0: self.clear_all()
		elif ip is not None: self.execute_instruction(ip, left_operand, right_operand)
		else: print(f"Non-instruction line: {line}")
		self.prev_line_tokens = self.tokenize(line)

	def extract_instruction(self, line):
		line_words = line.split(" ")
		# If we are not conformant to the line syntax, this is a no-instruction line, so anytime a test fails, we return (None, None, None)
		if len(line_words) != 3: return (None, None, None)
		(left, mid, right) = (line_words[0], line_words[1], line_words[2])
		if mid not in ["<<", ">>"]: return (None, None, None)
		if not (len(left) > 0 and len(right) > 0): return (None, None, None)
		(l_initial, r_initial) = (left[0], right[0])
		if l_initial not in self.v_initial or r_initial not in self.v_initial: return (None, None, None)
		instruction_pattern = f"{l_initial} {mid} {r_initial}"
		if not (left in self.v_dict and right in self.v_dict): return (None, None, None)
		(l_operand, r_operand) = (self.v_dict[left], self.v_dict[right])
		return (instruction_pattern, l_operand, r_operand)

	def execute_instruction(self, ip, l, r):
		""" ip: instruction pattern, l: left operand, r: right operand """
		print(f"Executing instruction of type: {ip}")

		if ip == "s << m": self.elicit(r, l) # l is string, r is model
		elif ip == "m << s": self.reveal(l, r) # l is model, r is string
		elif ip == "s >> m": self.add_xent(r, l) # l is string, r is model
		elif ip == "m >> s": self.sub_xent(l, r) # l is model, r is string
		elif ip == "s << s": self.cat(l, r) # l is destination string, r is source string
		elif ip == "s >> s": self.cut(l, r) # l is source string, r is destination string
		elif ip == "m << m": self.reward(l, r) # l is player, r is judge
		elif ip == "m >> m": self.ensure(r, l) # l is judge, r is player

	def elicit(self, m, s):
		num_tokens = min(max(len(s), 1), self.max_len - len(s))
		response = self.post_json(m.url, "/generate", {"tokens": m.context, "n": num_tokens})
		s.tokens += response["tokens"]
		m.context += response["tokens"]
		print(f"Elicited {num_tokens}. s={self.format_tokens(s.tokens)}")

	def reveal(self, m, s):
		m.context += s.tokens
		print(f"Revealed {self.format_tokens(s.tokens)} to model")

	def add_xent(self, m, s):
		response = self.post_json(m.url, "/xent", {"tokens": s.tokens})
		m.xent_acc += response["xent"]
		print(f"{m.xent_acc=}")

	def sub_xent(self, m, s):
		response = self.post_json(m.url, "/xent", {"tokens": s.tokens})
		m.xent_acc -= response["xent"]
		print(f"{m.xent_acc=}")

	def cat(self, sl, sr):
		prev_line_padding = self.tokenize('\n') if len(sl) > 0 else []
		rhs = sr.tokens if len(sr) > 0 else (prev_line_padding + self.prev_line_tokens)
		sl.tokens = sl.tokens + rhs[:min(len(rhs), self.max_len - len(sl))]
		print(f"Cat: sl={self.format_tokens(sl.tokens)} rhs={self.format_tokens(rhs)}")

	def cut(self, sl, sr): # s >> s will simply erase s
		sr.tokens = sl.tokens[:min(len(sl), len(sr))]
		sl.tokens = sl.tokens[len(sr.tokens):]
		print(f"Cut: sl={self.format_tokens(sl.tokens)} sr={self.format_tokens(sr.tokens)}")

	def reward(self, mp, mj):
		mp.score_acc += mj.xent_acc
		mj.xent_acc = 0.0
		print(f"Reward: {mp.score_acc=}")

	def ensure(self, mp, mj):
		mp.score_acc += self.ensure_nonlinearity(mj.xent_acc)
		mj.xent_acc = 0.0
		print(f"Ensure: {mp.score_acc=}")

	def clear_all(self):
		for m in self.m_dict.values(): self.clear_model(m)
		for s in self.s_dict.values(): self.clear_string(s)
		self.prev_line_tokens = []
		print("Cleared all")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--llm_server_url", default=DEFAULT_LLM_SERVER_URL)
	parser.add_argument("--game_path", default=DEFAULT_GAME_PATH)
	parser.add_argument("--print_strings", action="store_true")
	args = parser.parse_args()
	env = Env(print_strings=args.print_strings)
	for m in env.m_dict.values():
		m.url = args.llm_server_url
	with open(args.game_path) as f:
		env.run(f.read())
