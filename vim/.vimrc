" Configuration file for vim
set modelines=0" CVE-2007-2438

" Normally we use vim-extensions. If you want true vi-compatibility
" remove change the following statements
set nocompatible" Use Vim defaults instead of 100% vi compatibility
filetype off "required

" install bundle
" 1) git clone https://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim
" 2) PluginInstall
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()

" let Vundle manage Vundle, required
Plugin 'gmarik/Vundle.vim'
"Plugin 'fatih/vim-go'
Plugin 'taglist.vim'
Plugin 'scrooloose/nerdtree.git'
Bundle 'jistr/vim-nerdtree-tabs.git'
"Plugin 'ajmwagar/vim-deus.git'
Plugin 'MarcWeber/vim-addon-mw-utils'

" All of your Plugins must be added before the following line
call vundle#end()            " required

autocmd Filetype tex setl updatetime=0
let g:tex_flavor='latex'
let g:livepreview_previewer = 'open -a Preview'
let g:livepreview_cursorhold_recompile = 0
let g:vimtex_quickfix_mode=0
set conceallevel=1
let g:tex_conceal='abdmg'

filetype plugin indent on    " required

set backspace=2" more powerful backspacing

syntax on
" 语法高亮

autocmd InsertLeave * se nocul
autocmd InsertEnter * se cul
" 用浅色高亮当前行

set smartindent
" 智能对齐

set autoindent
" 自动对齐

set confirm
" 在处理未保存或只读文件的时候，弹出确认

set tabstop=4
" Tab键的宽度

set softtabstop=4
set shiftwidth=4
"  统一缩进为4

set noexpandtab
" 不要用空格代替制表符

set number
" 显示行号

set history=50
" 历史纪录数

set hlsearch
set incsearch
" 搜索逐字符高亮

set gdefault
" 行内替换

set encoding=utf-8
set fileencodings=utf-8,ucs-bom,shift-jis,gb18030,gbk,gb2312,cp936,utf-16,big5,euc-jp,latin1
" 编码设置

" 设置颜色主题
colorscheme deus
"hi StatusLine ctermbg=grey
hi StatusLine ctermbg=24 ctermfg=254 guibg=#004f87 guifg=#e4e4e4

"set guifont=Menlo:h16:cANSI
" 设置字体

set langmenu=zn_CN.UTF-8
set helplang=cn
" 语言设置

set cmdheight=2
" 命令行（在状态行）的高度，默认为1,这里是2

set ruler
" 在编辑过程中，在右下角显示光标位置的状态行

set laststatus=2
" 总是显示状态行

set showcmd
" 在状态行显示目前所执行的命令，未完成的指令片段亦会显示出来

set scrolloff=3
" 光标移动到buffer的顶部和底部时保持3行距离

set showmatch
" 高亮显示对应的括号 
set matchtime=5
" 对应括号高亮的时间（单位是十分之一秒）

set autowrite
" 在切换buffer时自动保存当前文件

set wildmenu
" 增强模式中的命令行自动完成操作
" 增强模式中的命令行自动完成操作
set linespace=2
" 字符间插入的像素行数目

set whichwrap=b,s,<,>,[,]
" 开启Normal或Visual模式下Backspace键，空格键，左方向键，右方向键，Insert或replace模式下左方向键，右方向键跳行的功能。

filetype plugin indent on
" 分为三部分命令：file on, file plugin on, file indent on.分别表示自动识别文件类型，用文件类型脚本，使用缩进定义文件。

let Tlist_Ctags_Cmd='ctags'
let Tlist_Show_One_File=1               "不同时显示多个文件的tag，只显示当前文件的
let Tlist_WinWidt =28                    "设置taglist的宽度
let Tlist_Exit_OnlyWindow=1             "如果taglist窗口是最后一个窗口，则退出vim
let Tlist_Ctags_Cmd="/usr/local/bin/ctags"
"let Tlist_Use_Right_Window=1            "在右侧窗口中显示taglist窗口
"let Tlist_Use_Left_Windo =1                "在左侧窗口中显示taglist窗口


"==================自定义的键映射======================

vnoremap $1 <esc>`>a)<esc>`<i(<esc>
vnoremap $2 <esc>`>a]<esc>`<i[<esc>
vnoremap $3 <esc>`>a}<esc>`<i{<esc>
vnoremap $$ <esc>`>a"<esc>`<i"<esc>
vnoremap $q <esc>`>a'<esc>`<i'<esc>
vnoremap $e <esc>`>a"<esc>`<i"<esc>

map<M-Up> :m .+1<CR>==
nnoremap <A-k> :m .-2<CR>==
inoremap <C-j> <Esc>:m .+1<CR>==gi
inoremap <A-k> <Esc>:m .-2<CR>==gi
vnoremap <A-j> :m '>+1<CR>gv=gv
vnoremap <A-k> :m '<-2<CR>gv=gv

" 括号自动生成

map <F7> :if exists("syntax_on") <BAR>
\    syntax off <BAR><CR>
\  else <BAR>
\syntax enable <BAR>
\  endif
" 单键<F7>控制syntax on/off。原因是有时候颜色太多会妨碍阅读。

map <F5> :call CompileRunGcc()<CR>
func! CompileRunGcc()
exec "w"
exec "!gcc % -o %<"
exec "! ./%<"
endfunc
" <F5>编译和运行C程序

map <F6> :call CompileRunGpp()<CR>
func! CompileRunGpp()
	if &filetype=='cpp'
		exec "w"
		exec "!g++ % -o %<"
		exec "! ./<"
	elseif &filetype=='python'
		exec "!time python %"
	endif
endfunc
" <F6>编译和运行C++程序

" Don't write backup file if vim is being called by "crontab -e"
au BufWrite /private/tmp/crontab.* set nowritebackup
" Don't write backup file if vim is being called by "chpass"
au BufWrite /private/etc/pw.* set nowritebackup
