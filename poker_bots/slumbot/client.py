import requests
import sys

HOST = 'slumbot.com'
NUM_STREETS = 4
SMALL_BLIND = 50
BIG_BLIND = 100
STACK_SIZE = 20000

class SlumbotClient:
    def __init__(self, username=None, password=None):
        self.token = None
        if username and password:
            self.login(username, password)

    def login(self, username, password):
        data = {"username": username, "password": password}
        response = requests.post(f'https://{HOST}/slumbot/api/login', json=data)
        if response.status_code != 200:
            raise Exception(f"Login failed: {response.status_code} {response.text}")
        
        r = response.json()
        if 'error_msg' in r:
            raise Exception(f"Login error: {r['error_msg']}")
            
        self.token = r.get('token')
        print(f"Logged in, token: {self.token}")

    def new_hand(self):
        data = {}
        if self.token:
            data['token'] = self.token
            
        response = requests.post(f'https://{HOST}/slumbot/api/new_hand', json=data)
        if response.status_code != 200:
            raise Exception(f"NewHand failed: {response.status_code} {response.text}")
            
        r = response.json()
        if 'error_msg' in r:
            raise Exception(f"NewHand error: {r['error_msg']}")
            
        # Update token if present
        if r.get('token'):
            self.token = r.get('token')
            
        return r

    def act(self, action_str):
        if not self.token:
            raise Exception("No token available for Act")
            
        data = {'token': self.token, 'incr': action_str}
        response = requests.post(f'https://{HOST}/slumbot/api/act', json=data)
        if response.status_code != 200:
            raise Exception(f"Act failed: {response.status_code} {response.text}")
            
        r = response.json()
        if 'error_msg' in r:
            raise Exception(f"Act error: {r['error_msg']}")
            
        if r.get('token'):
            self.token = r.get('token')
            
        return r

    @staticmethod
    def parse_action(action):
        """
        Parses the action string from Slumbot.
        Returns state dict.
        """
        st = 0
        street_last_bet_to = BIG_BLIND
        total_last_bet_to = BIG_BLIND
        last_bet_size = BIG_BLIND - SMALL_BLIND
        last_bettor = 0 # 0=BB (pos 0), 1=SB (pos 1). Wait.
        # "A client_pos of 0 indicates that you are the big blind ... 1 indicates you are the small blind."
        # Preflop: SB(1) posts 50, BB(0) posts 100.
        # First to act Preflop is SB(1).
        # Wait, usually SB acts first Heads Up.
        # Slumbot Doc: "0 indicates ... Big Blind (second to act preflop)". 
        # So SB is 1. SB acts first.
        # last_bettor = 0 means BB posted 100.
        
        sz = len(action)
        pos = 1 # SB starts
        
        if sz == 0:
            return {
                'st': st,
                'pos': pos,
                'street_last_bet_to': street_last_bet_to,
                'total_last_bet_to': total_last_bet_to,
                'last_bet_size': last_bet_size,
                'last_bettor': last_bettor,
            }

        check_or_call_ends_street = False
        i = 0
        while i < sz:
            if st >= NUM_STREETS:
                return {'error': 'Unexpected error'}
            c = action[i]
            i += 1
            if c == 'k':
                if last_bet_size > 0:
                    return {'error': 'Illegal check'}
                if check_or_call_ends_street:
                    if st < NUM_STREETS - 1 and i < sz:
                        if action[i] != '/': return {'error': 'Missing slash'}
                        i += 1
                    if st == NUM_STREETS - 1: pos = -1
                    else:
                        pos = 0 # Postflop, BB(0) acts first?
                        # "first to act postflop" is BB(0).
                        st += 1
                    street_last_bet_to = 0
                    check_or_call_ends_street = False
                else:
                    pos = (pos + 1) % 2
                    check_or_call_ends_street = True
            elif c == 'c':
                if last_bet_size == 0: return {'error': 'Illegal call'}
                if total_last_bet_to == STACK_SIZE:
                    # All in call
                    if i != sz:
                        # Skip remaining slashes
                        while i < sz:
                            if action[i] == '/': i+=1
                            else: return {'error': 'Extra chars'}
                    st = NUM_STREETS - 1
                    pos = -1
                    last_bet_size = 0
                    return {
                        'st': st, 'pos': pos, 
                        'street_last_bet_to': street_last_bet_to,
                        'total_last_bet_to': total_last_bet_to,
                        'last_bet_size': last_bet_size,
                        'last_bettor': last_bettor
                    }
                
                if check_or_call_ends_street:
                    if st < NUM_STREETS - 1 and i < sz:
                        if action[i] != '/': return {'error': 'Missing slash'}
                        i += 1
                    if st == NUM_STREETS - 1: pos = -1
                    else:
                        pos = 0
                        st += 1
                    street_last_bet_to = 0
                    check_or_call_ends_street = False
                else:
                    pos = (pos + 1) % 2
                    check_or_call_ends_street = True
                last_bet_size = 0
                last_bettor = -1
            elif c == 'f':
                pos = -1
                return {
                    'st': st, 'pos': pos, 
                    'street_last_bet_to': street_last_bet_to,
                    'total_last_bet_to': total_last_bet_to,
                    'last_bet_size': last_bet_size,
                    'last_bettor': last_bettor
                }
            elif c == 'b':
                j = i
                while i < sz and action[i] >= '0' and action[i] <= '9': i += 1
                if i == j: return {'error': 'Missing bet size'}
                new_street_last_bet_to = int(action[j:i])
                new_last_bet_size = new_street_last_bet_to - street_last_bet_to
                
                last_bet_size = new_last_bet_size
                street_last_bet_to = new_street_last_bet_to
                total_last_bet_to += last_bet_size
                last_bettor = pos
                pos = (pos + 1) % 2
                check_or_call_ends_street = True
            else:
                return {'error': 'Unexpected char'}

        return {
            'st': st,
            'pos': pos,
            'street_last_bet_to': street_last_bet_to,
            'total_last_bet_to': total_last_bet_to,
            'last_bet_size': last_bet_size,
            'last_bettor': last_bettor,
        }

