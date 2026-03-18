// AI智能客服系统前端JavaScript

class CustomerServiceApp {
    constructor() {
        this.apiBaseUrl = 'http://localhost:8080/api/v1';
        this.userId = 'guest_001';
        this.sessionId = this.generateSessionId();
        this.init();
    }

    init() {
        this.bindEvents();
        this.loadInitialData();
        console.log('AI智能客服系统前端初始化完成');
    }

    generateSessionId() {
        return 'sess_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    bindEvents() {
        // 消息发送事件
        const messageInput = document.getElementById('messageInput');
        if (messageInput) {
            messageInput.addEventListener('keypress', this.handleKeyPress.bind(this));
        }

        // 知识库搜索事件
        const knowledgeSearch = document.getElementById('knowledgeSearch');
        if (knowledgeSearch) {
            knowledgeSearch.addEventListener('input', this.debounce(this.searchKnowledge.bind(this), 300));
        }

        // 分类筛选事件
        const categoryFilter = document.getElementById('categoryFilter');
        if (categoryFilter) {
            categoryFilter.addEventListener('change', this.filterKnowledge.bind(this));
        }

        // 星级评分事件
        this.initStarRating();
    }

    handleKeyPress(event) {
        if (event.key === 'Enter') {
            this.sendMessage();
        }
    }

    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    async sendMessage(message = null) {
        const input = document.getElementById('messageInput');
        const messageText = message || input.value.trim();
        
        if (!messageText) return;
        
        if (!message) {
            input.value = '';
        }
        
        this.addMessageToUI(messageText, 'user');
        const loadingMessageId = this.addMessageToUI('正在思考...', 'ai', true);
        
        try {
            const response = await fetch(`${this.apiBaseUrl}/chat/message`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_id: this.userId,
                    message: messageText,
                    session_id: this.sessionId
                })
            });
            
            const data = await response.json();
            this.removeMessageFromUI(loadingMessageId);
            
            if (data.success) {
                this.addMessageToUI(data.data.reply, 'ai');
                if (data.data.recommendations) {
                    this.showRecommendations(data.data.recommendations);
                }
            } else {
                this.addMessageToUI('抱歉，我现在无法回答这个问题，请稍后再试。', 'ai');
            }
            
        } catch (error) {
            console.error('发送消息失败:', error);
            this.removeMessageFromUI(loadingMessageId);
            this.addMessageToUI('网络连接出现问题，请检查网络后重试。', 'ai');
        }
    }

    addMessageToUI(content, sender, isLoading = false) {
        const chatMessages = document.getElementById('chatMessages');
        if (!chatMessages) return;
        
        const messageId = 'msg_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        messageDiv.id = messageId;
        
        const avatarIcon = sender === 'user' ? 'fas fa-user' : 'fas fa-robot';
        const timeStr = new Date().toLocaleTimeString('zh-CN', { 
            hour: '2-digit', minute: '2-digit' 
        });
        
        messageDiv.innerHTML = `
            <div class="message-content">
                <i class="${avatarIcon} me-2"></i>${content}
                ${isLoading ? '<span class="loading ms-2"></span>' : ''}
            </div>
            <div class="message-time">${timeStr}</div>
        `;
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        return isLoading ? messageId : null;
    }

    removeMessageFromUI(messageId) {
        const element = document.getElementById(messageId);
        if (element) element.remove();
    }

    async loadRecommendations() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/recommendations/user/${this.userId}?top_k=6`);
            const data = await response.json();
            if (data.success) {
                this.renderRecommendations(data.data);
            }
        } catch (error) {
            console.error('加载推荐数据失败:', error);
        }
    }

    renderRecommendations(recommendations) {
        const container = document.getElementById('recommendationsList');
        if (!container) return;
        
        container.innerHTML = '';
        recommendations.items.slice(0, 6).forEach((itemId, index) => {
            const col = document.createElement('div');
            col.className = 'col-md-4 mb-3';
            col.innerHTML = `
                <div class="card recommendation-card">
                    <div class="card-img-top bg-light d-flex align-items-center justify-content-center" style="height: 150px;">
                        <i class="fas fa-${['laptop', 'headphones', 'mobile-alt'][index % 3]} fa-3x text-${['primary', 'info', 'success'][index % 3]}"></i>
                    </div>
                    <div class="card-body">
                        <h6 class="card-title">推荐商品 ${index + 1}</h6>
                        <p class="card-text small">基于您的偏好精心推荐</p>
                        <div class="d-flex justify-content-between align-items-center">
                            <span class="text-warning">${'★'.repeat(Math.floor((recommendations.scores[index] || 0.8) * 5))}${'☆'.repeat(5-Math.floor((recommendations.scores[index] || 0.8) * 5))}</span>
                            <span class="badge bg-success">推荐</span>
                        </div>
                    </div>
                </div>
            `;
            container.appendChild(col);
        });
    }

    initStarRating() {
        const stars = document.querySelectorAll('#starRating .star');
        stars.forEach(star => {
            star.addEventListener('click', (e) => {
                const rating = parseInt(e.target.dataset.rating);
                this.setStarRating(rating);
            });
        });
    }

    setStarRating(rating) {
        const stars = document.querySelectorAll('#starRating .star');
        stars.forEach((star, index) => {
            if (index < rating) {
                star.classList.remove('far');
                star.classList.add('fas', 'active');
            } else {
                star.classList.remove('fas', 'active');
                star.classList.add('far');
            }
        });
    }

    loadInitialData() {
        this.loadRecommendations();
        this.loadKnowledgeBase();
        this.loadAnalytics();
    }

    loadKnowledgeBase() {
        // 模拟数据
        const mockData = [
            { title: '产品使用入门指南', category: '产品说明', date: '2024-01-15', desc: '详细介绍产品的初始设置和基本使用方法...' },
            { title: '常见问题解答', category: '常见问题', date: '2024-01-10', desc: '收集用户最常遇到的问题和解决方案...' }
        ];
        this.renderKnowledgeBase(mockData);
    }

    renderKnowledgeBase(articles) {
        const container = document.getElementById('knowledgeResults');
        if (!container) return;
        
        container.innerHTML = '';
        articles.forEach(article => {
            const col = document.createElement('div');
            col.className = 'col-md-6 mb-3';
            col.innerHTML = `
                <div class="card knowledge-card">
                    <div class="card-body">
                        <h6 class="card-title">${article.title}</h6>
                        <p class="card-text text-muted small">${article.desc}</p>
                        <div class="d-flex justify-content-between align-items-center">
                            <span class="badge bg-primary">${article.category}</span>
                            <small class="text-muted">${article.date}</small>
                        </div>
                    </div>
                </div>
            `;
            container.appendChild(col);
        });
    }

    loadAnalytics() {
        // 初始化图表占位符
        console.log('分析数据加载完成');
    }
}

// 全局函数
function sendMessage() {
    app.sendMessage();
}

function sendQuickMessage(message) {
    app.sendMessage(message);
}

function handleKeyPress(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}

function clearChat() {
    const chatMessages = document.getElementById('chatMessages');
    if (chatMessages) {
        chatMessages.innerHTML = '<div class="message system-message"><div class="message-content"><i class="fas fa-robot me-2"></i>对话已清空，请问有什么可以帮助您的？</div><div class="message-time">刚刚</div></div>';
    }
}

function exportChat() {
    alert('聊天记录导出功能开发中...');
}

function showHelp() {
    alert('帮助中心：\n1. 输入问题开始对话\n2. 使用快捷按钮快速提问\n3. 查看知识库获取更多信息\n4. 在推荐中心发现个性化内容');
}

function submitFeedback() {
    alert('感谢您的反馈！');
    $('#feedbackModal').modal('hide');
}

// 初始化应用
const app = new CustomerServiceApp();