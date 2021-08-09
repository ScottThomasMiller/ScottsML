import SwiftUI

// forked from: https://stackoverflow.com/questions/58896661/swiftui-create-image-slider-with-dots-as-indicators

struct ContentView: View {
    let interval = 1.0
    @State private var timer = Timer.publish(every: 1.0, on: .main, in: .common).autoconnect()
    @State private var selection = 0
    @State var isTimerRunning = true
    let images: [LabeledImage] = prepareImages()
    let outlet = Outlet(streamInfo: StreamInfo(name: "ImageLabel",
                                               format: ChannelFormat(format: lsl_channel_format_t(3)),
                                               id: "imageType",
                                               sampleRate: 1.0))

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
                    print("selection: \(selection) label: \(images[selection+1].label)")
                    do {
                        try outlet.push(data: images[selection].label)
                    }
                    catch {
                        print("Cannot push to outlet. Error code: \(error)")
                    }
                    selection += 1
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
    }
    
    func stopTimer() {
        self.timer.upstream.connect().cancel()
    }
    
    func startTimer() {
        self.timer = Timer.publish(every: 1.0, on: .main, in: .common).autoconnect()
    }
}
