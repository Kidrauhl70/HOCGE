def FileDirect(args):
     if args.input == 'ship':
          if args.hon:
               args.input = 'C:/Users/Lenovo/Desktop/MyCode/data/dat_dge/ship2.txt'
               fon_filename = 'C:/Users/Lenovo/Desktop/MyCode/data/dat_dge/ship1.txt'
          else:
               args.input = 'C:/Users/Lenovo/Desktop/MyCode/data/dat_dge/ship1.txt'
               fon_filename = args.input
          if args.label_file:
               args.label_file = 'C:/Users/Lenovo/Desktop/MyCode/data/dat_dge/ship_labels.csv'
          else:
               args.label_file = None
     elif args.input == 'wiki':
          if args.hon:
               args.input = 'C:/Users/Lenovo/Desktop/MyCode/data/dat_dge/wiki2.txt'
               fon_filename = 'C:/Users/Lenovo/Desktop/MyCode/data/dat_dge/wiki1.txt'
          else:
               args.input = 'C:/Users/Lenovo/Desktop/MyCode/data/dat_dge/wiki1.txt'
               fon_filename = args.input
          if args.label_file:
               args.label_file = 'C:/Users/Lenovo/Desktop/MyCode/data/dat_dge/wiki_labels.csv'
          else:
               args.label_file = None
     elif args.input == 'wikijump':
          if args.hon:
               args.input = 'C:/Users/Lenovo/Desktop/MyCode/data/dat_dge/wikijump/wikijump_t15_order2.txt'
               fon_filename = 'C:/Users/Lenovo/Desktop/MyCode/data/dat_dge/wikijump/wikijump_order1.txt'
          else:
               args.input = 'C:/Users/Lenovo/Desktop/MyCode/data/dat_dge/wikijump/wikijump_order1.txt'
               fon_filename = args.input
          if args.label_file:
               args.label_file = 'C:/Users/Lenovo/Desktop/MyCode/data/dat_dge/wiki_labels.csv'
          else:
               args.label_file = None
     elif args.input == 'click':
          if args.hon:
               args.input = 'C:/Users/Lenovo/Desktop/MyCode/data/dat_dge/click2.txt'
               fon_filename = 'C:/Users/Lenovo/Desktop/MyCode/data/dat_dge/click1.txt'
          else:
               args.input = 'C:/Users/Lenovo/Desktop/MyCode/data/dat_dge/click1.txt'
               fon_filename = args.input
          if args.label_file:
               args.label_file = 'C:/Users/Lenovo/Desktop/MyCode/data/dat_dge/click_labels.csv'
          else:
               args.label_file = None
     elif args.input == 'traces':
          if args.hon:
               args.input = 'C:/Users/Lenovo/Desktop/MyCode/data/dat_dge/traces3.txt'
               fon_filename = 'C:/Users/Lenovo/Desktop/MyCode/data/dat_dge/traces1.txt'
          else:
               args.input = 'C:/Users/Lenovo/Desktop/MyCode/data/dat_dge/traces1.txt'
               fon_filename = args.input
          if args.label_file:
               args.label_file = 'C:/Users/Lenovo/Desktop/MyCode/data/dat_dge/traces_labels.csv'
          else:
               args.label_file = None
     elif args.input == 'modify':
          if args.hon:
               args.input = 'C:/Users/Lenovo/Desktop/MyCode/data/dat_dge/modify2.txt'
               fon_filename = 'C:/Users/Lenovo/Desktop/MyCode/data/dat_dge/modify1.txt'
          else:
               args.input = 'C:/Users/Lenovo/Desktop/MyCode/data/dat_dge/modify1.txt'
               fon_filename = args.input
          if args.label_file:
               args.label_file = 'C:/Users/Lenovo/Desktop/MyCode/data/dat_dge/modify_labels.csv'
          else:
               args.label_file = None
     elif args.input == 'book':
          if args.hon:
               args.input = 'C:/Users/Lenovo/Desktop/MyCode/data/dat_dge/book2.txt'
               fon_filename = 'C:/Users/Lenovo/Desktop/MyCode/data/dat_dge/book1.txt'
          else:
               args.input = 'C:/Users/Lenovo/Desktop/MyCode/data/dat_dge/book1.txt'
               fon_filename = args.input
          if args.label_file:
               args.label_file = 'C:/Users/Lenovo/Desktop/MyCode/data/dat_dge/book_labels.csv'
          else:
               args.label_file = None                
     else:
          if args.hon:
               args.input = 'C:/Users/Lenovo/Desktop/MyCode/data/dat_dge/air2.txt'
               fon_filename = 'C:/Users/Lenovo/Desktop/MyCode/data/dat_dge/air1.txt'
          else:
               args.input = 'C:/Users/Lenovo/Desktop/MyCode/data/dat_dge/air1.txt'
               fon_filename = args.input
          if args.label_file:
               args.label_file = 'C:/Users/Lenovo/Desktop/MyCode/data/dat_dge/air_labels.csv'
          else:
               args.label_file = None

     return args.input, fon_filename, args.label_file