import SwiftUI

// forked from: https://stackoverflow.com/questions/58896661/swiftui-create-image-slider-with-dots-as-indicators

struct ContentView: View {
    let interval = 1.0
    @State private var timer = Timer.publish(every: 1.0, on: .main, in: .common).autoconnect()
    @State private var selection = -1
    @State var isTimerRunning = false
    let images: [LabeledImage] = prepareImages()
    let outlet = Outlet(streamInfo: StreamInfo(name: "ImageLabel",
                                               format: .string,
                                               id: "imageType",
                                               sampleRate: LSL_IRREGULAR_RATE))


    var body: some View {
        ZStack{
            Color.black
            TabView(selection : $selection){
                ForEach(0..<images.count){ i in
                    Image(uiImage: images[i].image)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                }
            }
            .tabViewStyle(PageTabViewStyle())
            .indexViewStyle(PageIndexViewStyle(backgroundDisplayMode: .always))
            .onReceive(timer, perform: { _ in
                withAnimation{
                    guard selection < (images.count-1) else {
                        print("done")
                        self.stopTimer()
                        return
                    }
                    if selection < 0 {
                        print("pause")
                        selection = 0
                        self.stopTimer()
                    } else {
                        let label = images[selection+1].label
                        let timestamp = lsl_local_clock()
                        print("label: \(label) timestamp: \(timestamp)")
                        do {
                            try outlet.push(data: label)
                            //try outlet.push(data: labels)
                        }
                        catch {
                            print("Cannot push to outlet. Error code: \(error)")
                        }
                        selection += 1
                    }
                }
            })
            .animation(nil)
            .onTapGesture {
                if isTimerRunning {
                    self.stopTimer()
                } else {
                    self.startTimer()
                }
                isTimerRunning.toggle()
            }
        }
    }
    
    init() {
        print("total images: \(images.count)")
        let eeg = EEG()
        DispatchQueue.global(qos: .background).async {
            eeg.streamToFile()
        }
    }
    
    func stopTimer() {
        print("stop timer")
        self.timer.upstream.connect().cancel()
    }
    
    func startTimer() {
        print("start timer")
        self.timer = Timer.publish(every: 1.0, on: .main, in: .common).autoconnect()
    }
}
