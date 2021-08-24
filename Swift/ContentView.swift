//
//  ContentView.swift
//  BrainBook
//
//  Created by Scott Miller on 8/23/21.
//

import SwiftUI

func testBrainFlow() {
    let params = BrainFlowInputParams(serial_port: "/dev/cu.usbserial-DM0258EJ")
    let board = BoardShim(BoardIds.CYTON_DAISY_BOARD, params)
    do {
        try board.prepareSession()
        try board.isPrepared() ? print("open") : print("closed")
        
        try board.releaseSession()
        try board.isPrepared() ? print("open") : print("closed")
    } catch {
        print("Error: \(error)")
    }
}

struct ContentView: View {
    var body: some View {
        testBrainFlow()
        return Text("Hello, world!")
            .padding()
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
