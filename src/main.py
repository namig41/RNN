#! /usr/bin/python3 

from display import Display

def main():
    app = Display('hand.avi', 840, 640)
    app.run()

if __name__ == "__main__":
    main() 
