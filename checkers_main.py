
# ------------------------------------------------------------
# Demo
# ------------------------------------------------------------
from checkers_board import BoardPosition,HexBoard
from checkers_pins import Pin
from checkers_gui import BoardGUI
import numpy as np
import re

running = True
num_players = 0
assigned = []
half_colours1 = ['red', 'lawn green', 'yellow']
half_colours2 = ['blue', 'gray0', 'purple']

if __name__ == "__main__":
    board = HexBoard(R=4, hole_radius=16, spacing=34)
    boardPins = []
    num_turns = 0
    turn = ['']
    input_message = "Type 'exit' to exit. Type 'assign' and press enter for first player: "
    help_mode = True

    

    while running==True:
        command = input(input_message)
        if command == 'exit':
            exit()
        if command == 'show':
            gui.run()
        if command == 'assign' and num_players %2 == 0 and num_turns==0:
            available_colours = [c for c in half_colours1+half_colours2 if c not in assigned]
            print('Av:',available_colours)
            choice = np.random.choice(available_colours)
            print('Assigning:',choice)
            num_players+=1
            colour_set=choice
            assigned.append(choice)

            axials_colour = board.axial_of_colour(colour_set)
            pins = []   
            pins = [Pin(board, axials_colour[i], id=i, color=colour_set) for i in range(10)]
            boardPins+= pins
            print(boardPins)

            print("\nASCII hex board (R=4):\n")
            board.print_ascii(pins=boardPins, empty='·')
            input_message = "Type 'assign' and press enter for second player: "
            continue
        if command == 'assign' and num_players % 2 ==1 and num_turns==0:
            choice = board.colour_opposites[assigned[-1]]
            print('Assigning', choice)
            num_players+=1
            colour_set=choice
            assigned.append(choice)

            axials_colour = board.axial_of_colour(colour_set)
            pins = []   
            pins = [Pin(board, axials_colour[i], id=i, color=colour_set) for i in range(10)]
            boardPins+= pins

            print("\nASCII hex board (R=4):\n")
            board.print_ascii(pins=boardPins, empty='·')
            input_message = "Type 'assign' and press enter for more players, else type 'start game': "
            continue
        '''if command == 'assign' and num_players > 2 and num_turns==0:
            temp = [h for h in half_colours1+half_colours2 if h not in assigned]
            choice = np.random.choice(temp)
            print('Assigning', choice)
            num_players+=1
            colour_set=choice
            assigned.append(choice)

            axials_colour = board.axial_of_colour(colour_set)
            pins = []   
            pins = [Pin(board, axials_colour[i], id=i, color=colour_set) for i in range(10)]
            boardPins+= pins

            print("\nASCII hex board (R=4):\n")
            board.print_ascii(pins=boardPins, empty='·')
            input_message = "Type 'assign' and press enter for more players, else type 'start game': "
            continue'''
        if command == 'start game' and num_turns==0:
            gui = BoardGUI(board, boardPins)
            candidate_turn = num_turns%len(assigned)
            turn[0] = assigned[candidate_turn]
            turn_pins_positions = [(pin.id, pin.axialindex) for pin in boardPins if pin.color==turn[-1]]
            print(assigned[candidate_turn].upper() +"'s Turn")
            print(assigned[candidate_turn] +"'s current positions : format (pin_id, pin_axialindex)\n"+str(turn_pins_positions))
            if help_mode:
                helping = True
                while helping:
                    pin_try_inp = input("Helpmode:Which pin will you move? " )
                    if pin_try_inp == 'exit':
                        exit()
                    try_Pin = [pin for pin in boardPins if pin.color==turn[0] and str(pin.id) ==pin_try_inp][0]
                    print("Possible moves:", try_Pin.getPossibleMoves())
                    help_continue = input("Need more help? Yes/No ")
                    if 'Yes'==help_continue:
                        continue
                    elif 'No'==help_continue:
                        helping = False
                        continue
                    else:
                        continue
            input_message = 'Enter '+assigned[candidate_turn] +"'s move: (pin_number, dest_axial): "
            continue
        if "'s move: " in input_message:
            if command == 'exit':
                exit()
            regex = r"\(\d+,\d+\)"
            if not re.match(regex, command):
                print('Format wrong, needs (pin_number, pin_axialposition)')
                turn_success = False
            else:
                pin_num = command.split(',')[0].replace('(','')
                dest = int(command.split(',')[1].replace(')',''))
                turn_Pin = [pin for pin in boardPins if pin.color==turn[0] and str(pin.id) ==pin_num][0]
                turn_success = turn_Pin.placePin(dest)
            print(turn_success)
            if turn_success:
                turn_pins_positions = [(pin.id, pin.axialindex) for pin in boardPins if pin.color==turn[-1]]
                print(assigned[candidate_turn].upper() +"'s Turn")
                print(assigned[candidate_turn] +"'s current positions : format (pin_id, pin_axialindex)\n"+str(turn_pins_positions))
                print("\nASCII hex board (R=4):\n")
                board.print_ascii(pins=boardPins, empty='·')
                gui.refresh(boardPins)
                num_turns+=1
                candidate_turn = num_turns%len(assigned)
                turn[0] = assigned[candidate_turn]
                turn_pins_positions = [(pin.id, pin.axialindex) for pin in boardPins if pin.color==turn[-1]]
                print(assigned[candidate_turn].upper() +"'s Turn")
                print(assigned[candidate_turn] +"'s current positions : format (pin_id, pin_axialindex)\n"+str(turn_pins_positions))
                if help_mode:
                    helping = True
                    while helping:
                        pin_try_inp = input("Helpmode:Which pin will you move? " )
                        if pin_try_inp == 'exit':
                            exit()
                        try_Pin = [pin for pin in boardPins if pin.color==turn[0] and str(pin.id) ==pin_try_inp][0]
                        print("Possible moves:", try_Pin.getPossibleMoves())
                        help_continue = input("Need more help? Yes/No ")
                        if 'Yes'==help_continue:
                            continue
                        elif 'No'==help_continue:
                            helping = False
                            continue
                        else:
                            continue
                input_message = "Enter "+assigned[candidate_turn] +"'s move: (pin_number, dest_axial): "
            else:
                print(assigned[candidate_turn] +" Try Again")
                turn_pins_positions = [(pin.id, pin.axialindex) for pin in boardPins if pin.color==turn[-1]]
                print(assigned[candidate_turn] +"'s current positions : format (pin_id, pin_axialindex)\n"+str(turn_pins_positions))
                if help_mode:
                    helping = True
                    while helping:
                        pin_try_inp = input("Helpmode:Which pin will you move? " )
                        if pin_try_inp == 'exit':
                            exit()
                        try_Pin = [pin for pin in boardPins if pin.color==turn[0] and str(pin.id) ==pin_try_inp][0]
                        print("Possible moves:", try_Pin.getPossibleMoves())
                        help_continue = input("Need more help? Yes/No ")
                        if 'Yes'==help_continue:
                            continue
                        elif 'No'==help_continue:
                            helping = False
                            continue
                        else:
                            continue
                input_message = "Enter "+assigned[candidate_turn] +"'s move: (pin_number, dest_axial): "
            continue








            







    '''colour_set='red'
    axials_colour = board.axial_of_colour(colour_set)
    print(axials_colour)
    pins = []   
    pins = [Pin(board, axials_colour[i], id=i, color=colour_set) for i in range(10)]

    print("\nASCII hex board (R=4):\n")
    board.print_ascii(pins=pins, empty='·')

    #board.axial 

    gui = BoardGUI(board, pins)
    gui.run()'''
